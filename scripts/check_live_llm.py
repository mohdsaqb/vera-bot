"""Is the deployed bot's wording layer actually working?

/v1/metadata reports the configured model name whether or not the key is valid,
so it cannot answer this. This can: push one context set, tick, and compare the
body the deployment returned against the body the deterministic engine produces
from the same inputs. Identical means every call fell back; different means the
model wrote it.

    python scripts/check_live_llm.py https://vera-bot-hc2f.onrender.com
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.compose import compose  # noqa: E402

CHALLENGE = Path(__file__).resolve().parents[2] / "magicpin-ai-challenge" / "dataset"
TRIGGER_ID = "trg_004_perf_dip_bharat"
NOW = "2026-05-01T10:00:00Z"


def post(base: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{base.rstrip('/')}{path}", method="POST",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.loads(response.read())


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "https://vera-bot-hc2f.onrender.com"

    categories = {}
    for path in sorted((CHALLENGE / "categories").glob("*.json")):
        data = json.load(open(path))
        for entry in data if isinstance(data, list) else [data]:
            categories[entry.get("slug") or entry.get("category_slug")] = entry
    merchants = {m["merchant_id"]: m
                 for m in json.load(open(CHALLENGE / "merchants_seed.json"))["merchants"]}
    triggers = {t["id"]: t
                for t in json.load(open(CHALLENGE / "triggers_seed.json"))["triggers"]}

    trigger = triggers[TRIGGER_ID]
    merchant = merchants[trigger["merchant_id"]]
    category = categories[merchant["category_slug"]]

    post(base, "/v1/teardown", {})
    for scope, context_id, payload in (
        ("category", category["slug"], category),
        ("merchant", merchant["merchant_id"], merchant),
        ("trigger", trigger["id"], trigger),
    ):
        post(base, "/v1/context", {"scope": scope, "context_id": context_id,
                                   "version": 1, "payload": payload,
                                   "delivered_at": NOW})
    actions = post(base, "/v1/tick",
                   {"now": NOW, "available_triggers": [TRIGGER_ID]}).get("actions", [])

    live = next((a.get("body", "") for a in actions
                 if a.get("trigger_id") == TRIGGER_ID), "")
    if not live:
        print("[FAIL] the deployment returned no message for this trigger")
        return 1

    local = compose(category, merchant, trigger, now=NOW)
    baseline = local.body if local and local.decision == "send" else ""

    print(f"deployment : {live}\n")
    print(f"deterministic: {baseline}\n")
    if live.strip() == baseline.strip():
        print("[LLM OFF] byte-identical to the deterministic body — every wording call")
        print("          fell back. Check GROQ_API_KEY in the Render dashboard.")
        return 1
    print("[LLM ON] the deployment's wording differs from the deterministic body,")
    print("         so the model wrote it and the grounding validator accepted it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
