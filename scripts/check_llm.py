#!/usr/bin/env python3
"""Verify that the configured model can actually word a message.

Makes ONE real call to the provider with a real `GenerationBrief`, then reports
what came back and whether it survived the grounding validator. Run this before
deploying: it is the only way to confirm a model id, a key and the structured-output
mode all work together, and it costs a single completion.

    export GROQ_API_KEY=gsk_...
    export LLM_PROVIDER=groq
    export LLM_MODEL=openai/gpt-oss-120b
    python scripts/check_llm.py

Exits 0 when the model produced a usable body, 1 otherwise. A failure here is not
fatal to the bot — every message would simply be written deterministically — but it
means the wording layer is not earning its keep.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings, llm_api_key  # noqa: E402
from engine.types import GenerationBrief  # noqa: E402
from services.llm_service import LLMWriter, render_prompt  # noqa: E402
from services.message_validator import validate_generated_body  # noqa: E402

# A real decision, of the kind the engine produces: two facts, one live offer,
# one ask, a clinical category voice.
BRIEF = GenerationBrief(
    purpose="outbound",
    category="Dentists",
    audience="merchant",
    recipient_name="Meera",
    salutation="Dr. Meera",
    tone="peer_clinical",
    language="en",
    reason="perf dip",
    facts=(
        "calls are down 50% over the last 7 days against a baseline of 12",
        "your click-through rate is 2.1% against the 3% dentists median — "
        "0.9 percentage points below",
    ),
    offer_title="Dental Cleaning @ ₹299",
    offer_is_live=True,
    ask="Want me to put Dental Cleaning @ ₹299 back in front of people?",
    cta="binary_yes_no",
    taboo_terms=("guaranteed", "best in city"),
    style_notes=(
        "You are Vera, writing to the business owner.",
        "Write peer to peer, clinician to clinician.",
        "Never imply a cure, a guarantee or a medical claim.",
    ),
    deterministic_body=(
        "Dr. Meera — calls are down 50% over the last 7 days against a baseline of 12. "
        "Your click-through rate is 2.1% against the 3% dentists median — 0.9 "
        "percentage points below. Want me to put Dental Cleaning @ ₹299 back in "
        "front of people?"
    ),
)


def main() -> int:
    settings = get_settings()
    print("\nConfiguration")
    print(f"  provider          : {settings.llm_provider or '(none)'}")
    print(f"  model             : {settings.llm_model}")
    print(f"  structured output : {settings.llm_structured_method}")
    print(f"  temperature       : {settings.llm_temperature}")
    print(f"  max tokens        : {settings.llm_max_tokens}")
    print(f"  timeout           : {settings.llm_timeout_seconds}s")
    print(f"  reasoning effort  : {settings.llm_reasoning_effort or '(not sent)'}")
    print(f"  reasoning format  : {settings.llm_reasoning_format or '(not sent)'}")
    print(f"  api key           : {'present' if llm_api_key() else 'MISSING'}")

    writer = LLMWriter(settings)
    reason = writer.unavailable_reason()
    if reason:
        print(f"\n[FAIL] the wording layer is not usable: {reason}")
        print("  Set LLM_PROVIDER=groq and GROQ_API_KEY, then run this again.")
        return 1

    print(f"\nPrompt sent ({len(render_prompt(BRIEF))} chars) — one call, no retries.")
    started = time.perf_counter()
    result = writer.write(BRIEF)
    elapsed = (time.perf_counter() - started) * 1000

    print("\nResult")
    print(f"  source   : {result.source}")
    print(f"  latency  : {elapsed:.0f}ms (provider reported {result.latency_ms}ms)")
    if result.problems:
        print(f"  problems : {', '.join(result.problems)}")

    if not result.used_llm:
        print("\n[FAIL] the model did not produce a usable body.")
        print("  The bot would still work — every message written deterministically —")
        print("  but the wording layer is doing nothing. Common causes:")
        print("    llm_error:BadRequestError      model id not served, or")
        print("                                   structured output unsupported:")
        print("                                   try LLM_STRUCTURED_METHOD=function_calling")
        print("    llm_error:AuthenticationError  bad key")
        print("    llm_malformed_response         empty or unparseable structure;")
        print("                                   raise LLM_MAX_TOKENS, or set")
        print("                                   LLM_REASONING_EFFORT=low")
        print("    unsupported_*                  the model invented a fact and was")
        print("                                   correctly rejected — see below")
        if result.problems and any(p.startswith(("unsupported", "multiple", "taboo"))
                                   for p in result.problems):
            print("\n  That last case is the grounding validator working. Re-run a few")
            print("  times: occasional rejections are healthy, always is a prompt issue.")
        return 1

    print(f"\n  body ({len(result.body)} chars):\n    {result.body}")
    print(f"\n  deterministic version, for comparison:\n    {BRIEF.deterministic_body}")
    problems = validate_generated_body(result.body, BRIEF)
    print(f"\n  re-validated: {'clean' if not problems else ', '.join(problems)}")
    print("\n[PASS] the model produced a grounded, usable body.")
    print("  Set the same LLM_* values on your deployment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
