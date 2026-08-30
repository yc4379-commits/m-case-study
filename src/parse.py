"""
LLM structured extraction.

Turns the clean text produced by `extract.py` into a validated
`ResumeExtraction`. Three things distinguish this from calling a model and
hoping for JSON.

**Schema-constrained output.** The Pydantic schema is converted to a JSON
Schema and passed as a tool definition, so the model must emit an object of
that exact shape. The field descriptions in `schema.py` travel with it, which
is why the instructions live there rather than in one long prompt: guidance
sits next to the field it governs and cannot drift out of sync with it.

**Validation with corrective retry.** A schema-shaped response can still be
wrong -- an invented enum value, a date in the wrong format, an "evidence"
quote that does not appear in the source. We validate, and on failure send the
specific errors back for one repair attempt rather than discarding the work or
accepting it silently.

**Evidence verification.** Every evidence quote is checked against the source
text. A quote the model composed rather than copied is the most dangerous
failure mode in this system: it looks like proof and is not. We do not delete
these; we flag them, because knowing the model paraphrased is information.

Results are cached on a hash of (text, model, schema) so re-running the
notebook costs nothing and stays reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import anthropic
from pydantic import ValidationError

from schema import ResumeExtraction

DEFAULT_MODEL = "claude-sonnet-5"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "parses"

SYSTEM_PROMPT = """\
You extract structured data from investment-industry resumes for a hedge fund \
business development team.

Rules that matter more than completeness:

1. Transcribe, do not embellish. If the resume does not state something, \
return null. A null is useful; an invented value is a liability.

2. Evidence must be VERBATIM. Every `evidence` field must be a substring of \
the resume text, copied exactly. Never paraphrase, summarise or reconstruct a \
quote. If you cannot find supporting text, return an empty string and set \
confidence to "low".

3. Classify by substance, not vocabulary. What someone DID outranks what they \
called it. "Python" in a skills list is not evidence of systematic investing; \
a backtested factor model is.

4. Section headings lie. Some resumes file work history under "ACADEMIC \
PROFILE" or "KEY PROJECTS". An entry naming an employer, a role and a \
duration is a position regardless of the heading above it.

5. You are given the document text only, never its filename. If the resume \
body contains no name, return null -- do not reconstruct one.

6. Report your own difficulties in `extraction_notes`. Missing name, absent \
dates, ambiguous structure, apparent gaps. This drives the confidence score \
shown to the user, so understating problems degrades the product.\
"""


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------

@dataclass
class ParseResult:
    source_file: str
    model: str
    extraction: Optional[ResumeExtraction] = None
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    from_cache: bool = False
    validation_errors: list[str] = field(default_factory=list)
    unverified_quotes: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.extraction is not None

    @property
    def cost_usd(self) -> float:
        """Approximate spend, for the cost table in the notebook.

        Rates are configured rather than hard-coded because published prices
        change; see PRICING below.
        """
        rate_in, rate_out = PRICING.get(self.model, (0.0, 0.0))
        return (self.input_tokens * rate_in + self.output_tokens * rate_out) / 1e6


# USD per million tokens (input, output). Verify against current published
# pricing before quoting these figures anywhere they matter.
PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-5": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}


# --------------------------------------------------------------------------
# Schema plumbing
# --------------------------------------------------------------------------

def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve $ref/$defs into a self-contained schema.

    Nested Pydantic models produce $defs and $ref. Inlining them keeps the
    tool definition portable across providers and makes the schema readable
    when printed in the notebook, which matters for a reviewer.
    """
    defs = schema.pop("$defs", {})

    def resolve(node: Any, depth: int = 0) -> Any:
        if depth > 20:
            return node
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                target = resolve(json.loads(json.dumps(defs.get(name, {}))), depth + 1)
                merged = {k: v for k, v in node.items() if k != "$ref"}
                return {**target, **merged}
            return {k: resolve(v, depth + 1) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v, depth + 1) for v in node]
        return node

    return resolve(schema)


def build_tool() -> dict[str, Any]:
    """Build the Anthropic tool definition from the Pydantic schema."""
    return {
        "name": "record_resume",
        "description": (
            "Record the structured contents of one resume. Every field must "
            "be grounded in the supplied text."
        ),
        "input_schema": _inline_refs(ResumeExtraction.model_json_schema()),
    }


# --------------------------------------------------------------------------
# Evidence verification
# --------------------------------------------------------------------------

def _loose(text: str) -> str:
    """Normalise for quote comparison: collapse whitespace, fold case.

    Deliberately forgiving about whitespace and punctuation, because our own
    extraction reflows lines. Not forgiving about words -- that is the point.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def verify_quotes(extraction: ResumeExtraction, source: str) -> list[str]:
    """Return evidence quotes that do not appear in the source text."""
    haystack = _loose(source)
    unverified: list[str] = []

    def check(label: str, quote: str) -> None:
        if not quote:
            return
        needle = _loose(quote)
        if len(needle) > 8 and needle not in haystack:
            unverified.append(f"{label}: {quote[:90]}")

    check("investment_approach", extraction.investment_approach.evidence)
    check("market_side", extraction.market_side.evidence)
    for item in extraction.primary_sectors:
        check(f"sector[{item.value}]", item.evidence)
    for item in extraction.asset_classes:
        check(f"asset_class[{item.value}]", item.evidence)
    check("team_leadership", extraction.team_leadership.evidence)
    for metric in extraction.stated_metrics:
        check(f"stated_metric[{metric.kind}]", metric.quote)
    return unverified


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def _cache_key(text: str, model: str, tool: dict[str, Any]) -> str:
    """Hash every input that can change the output.

    An earlier version hashed only (text, model, schema). Editing the system
    prompt therefore left the cache valid, and a prompt fix appeared to have
    no effect -- the pipeline was quietly replaying pre-fix results. A cache
    key must cover the prompt as well, or it silently defeats the experiment
    it is meant to make cheap.
    """
    digest = hashlib.sha256()
    digest.update(text.encode())
    digest.update(model.encode())
    digest.update(json.dumps(tool, sort_keys=True).encode())
    digest.update(SYSTEM_PROMPT.encode())
    return digest.hexdigest()[:16]


def parse_resume(
    text: str,
    source_file: str,
    *,
    client: Optional[anthropic.Anthropic] = None,
    model: str = DEFAULT_MODEL,
    max_attempts: int = 2,
    use_cache: bool = True,
) -> ParseResult:
    """Extract one resume into a validated ResumeExtraction."""
    client = client or anthropic.Anthropic()
    tool = build_tool()
    result = ParseResult(source_file=source_file, model=model)

    if not text.strip():
        result.error = "Empty source text; nothing to parse."
        return result

    cache_path = CACHE_DIR / f"{_cache_key(text, model, tool)}.json"
    if use_cache and cache_path.exists():
        payload = json.loads(cache_path.read_text())
        result.extraction = ResumeExtraction.model_validate(payload["extraction"])
        result.attempts = payload.get("attempts", 1)
        result.input_tokens = payload.get("input_tokens", 0)
        result.output_tokens = payload.get("output_tokens", 0)
        result.from_cache = True
        result.unverified_quotes = verify_quotes(result.extraction, text)
        return result

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Extract this resume using the record_resume tool.\n\n"
                f"<resume>\n{text}\n</resume>"
            ),
        }
    ]

    started = time.monotonic()
    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt
        try:
            response = client.messages.create(
                model=model,
                max_tokens=8000,
                system=SYSTEM_PROMPT,
                tools=[tool],
                tool_choice={"type": "tool", "name": "record_resume"},
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            result.error = f"{type(exc).__name__}: {exc}"
            break

        result.input_tokens += response.usage.input_tokens
        result.output_tokens += response.usage.output_tokens

        block = next(
            (b for b in response.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if block is None:
            result.error = "Model returned no tool_use block."
            break

        try:
            result.extraction = ResumeExtraction.model_validate(block.input)
            result.validation_errors = []
            break
        except ValidationError as exc:
            errors = [
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                for e in exc.errors()[:10]
            ]
            result.validation_errors = errors
            if attempt == max_attempts:
                result.error = "Validation failed after retries."
                break
            # Feed the specific failures back rather than retrying blindly.
            messages += [
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "is_error": True,
                            "content": (
                                "Schema validation failed:\n- "
                                + "\n- ".join(errors)
                                + "\n\nCall record_resume again, fixing only "
                                "these fields."
                            ),
                        }
                    ],
                },
            ]

    result.latency_s = round(time.monotonic() - started, 2)

    if result.extraction is not None:
        result.unverified_quotes = verify_quotes(result.extraction, text)
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "extraction": result.extraction.model_dump(),
                        "attempts": result.attempts,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
    return result
