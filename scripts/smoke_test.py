#!/usr/bin/env python3
"""Post-deployment smoke test: drive a running Vera against the real contract.

Runs the whole judge-shaped flow: warmup, tick, reply, suppression, a version
bump, the reply families, audience separation and per-endpoint latency: against
whatever base URL it is given. Use it locally before pushing and against the
public HTTPS URL afterwards.

    python scripts/smoke_test.py                              # http://127.0.0.1:8000
    python scripts/smoke_test.py https://vera.onrender.com    # a deployment

The challenge dataset supplies the payloads; point CHALLENGE_DIR at it if this
script is not sitting beside the `magicpin-ai-challenge` directory.

Exits non-zero if any check fails, so it can gate a deploy. It calls
`/v1/teardown` first, which wipes the target's state: never run it against an
instance in the middle of a judged slot.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

NOW = "2026-04-26T10:35:00Z"

# Per-call budgets from challenge-testing-brief.md §5.
BUDGETS_MS = {
    "/v1/healthz": 2_000,
    "/v1/metadata": 2_000,
    "/v1/context": 5_000,
    "/v1/tick": 10_000,
    "/v1/reply": 10_000,
}

DIP, BHARAT = "trg_004_perf_dip_bharat", "m_002_bharat_dentist_mumbai"
RECALL, MEERA = "trg_003_recall_due_priya", "m_001_drmeera_dentist_delhi"
# Counted per merchant across threads, so the auto-reply run needs a merchant the
# reply families above have not already written to.
AUTO_MERCHANT = "m_003_studio11_salon_hyderabad"

ACTION_FIELDS = {
    "conversation_id", "merchant_id", "customer_id", "send_as", "trigger_id",
    "template_name", "template_params", "body", "cta", "suppression_key", "rationale",
}

results: list[tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    results.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))


def challenge_dir() -> Path:
    override = os.environ.get("CHALLENGE_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent.parent / "magicpin-ai-challenge"


class Bot:
    """Minimal client. Stdlib only, so this runs anywhere Python does."""

    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def call(self, method: str, path: str, body: dict | None = None, timeout: int = 40):
        data = json.dumps(body).encode() if body is not None else None
        request = urlrequest.Request(
            f"{self.base}{path}", data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            response = urlrequest.urlopen(request, timeout=timeout)
            payload = json.loads(response.read().decode())
            return response.status, payload, (time.perf_counter() - started) * 1000
        except urlerror.HTTPError as error:
            return error.code, json.loads(error.read().decode()), (
                time.perf_counter() - started
            ) * 1000

    def push(self, scope: str, context_id: str, payload: dict, version: int = 1):
        return self.call("POST", "/v1/context", {
            "scope": scope, "context_id": context_id, "version": version,
            "payload": payload, "delivered_at": NOW,
        })

    def tick(self, trigger_ids: list[str], now: str = NOW):
        return self.call("POST", "/v1/tick", {"now": now, "available_triggers": trigger_ids})

    def reply(self, conversation_id: str, message: str, merchant_id: str, **extra):
        return self.call("POST", "/v1/reply", {
            "conversation_id": conversation_id, "merchant_id": merchant_id,
            "from_role": "merchant", "message": message, "received_at": NOW,
            "turn_number": 2, **extra,
        })


def load_dataset(root: Path) -> dict:
    categories = {
        json.loads(p.read_text())["slug"]: json.loads(p.read_text())
        for p in (root / "categories").glob("*.json")
    }
    def indexed(filename: str, container: str, key: str) -> dict:
        payload = json.loads((root / filename).read_text())
        return {item[key]: item for item in payload[container] if key in item}

    return {
        "categories": categories,
        "merchants": indexed("merchants_seed.json", "merchants", "merchant_id"),
        "customers": indexed("customers_seed.json", "customers", "customer_id"),
        "triggers": indexed("triggers_seed.json", "triggers", "id"),
    }


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    dataset_root = challenge_dir() / "dataset"
    if not dataset_root.is_dir():
        print(f"challenge dataset not found at {dataset_root}; set CHALLENGE_DIR")
        return 2

    data = load_dataset(dataset_root)
    cats, merch = data["categories"], data["merchants"]
    trigs, custs = data["triggers"], data["customers"]
    bot = Bot(base)

    print(f"\nVera smoke test against {bot.base}\n")
    bot.call("POST", "/v1/teardown")

    # --- the five endpoints -------------------------------------------------
    latencies: dict[str, list[float]] = {path: [] for path in BUDGETS_MS}
    status, health, ms = bot.call("GET", "/v1/healthz")
    latencies["/v1/healthz"].append(ms)
    check("healthz responds ok", status == 200 and health.get("status") == "ok", f"{ms:.0f}ms")
    check("healthz reports all four scopes",
          set(health.get("contexts_loaded", {})) == {"category", "merchant", "customer", "trigger"})

    status, metadata, ms = bot.call("GET", "/v1/metadata")
    latencies["/v1/metadata"].append(ms)
    check("metadata responds ok", status == 200, f"{ms:.0f}ms")
    check("metadata has exactly the contract fields", set(metadata) == {
        "team_name", "team_members", "model", "approach", "contact_email",
        "version", "submitted_at"})
    check("metadata carries no secret",
          not any("key" in str(k).lower() for k in metadata)
          and "gsk" not in json.dumps(metadata),
          f"model={metadata.get('model', '')[:40]}")
    check("team identity is not a placeholder",
          metadata.get("team_name") not in {"", "Vera Rebuild"}
          and metadata.get("contact_email") not in {"", "unset@example.com"},
          f"{metadata.get('team_name')} <{metadata.get('contact_email')}>")

    # --- warmup -------------------------------------------------------------
    for scope, source, key in [("category", cats, "slug"),
                               ("merchant", merch, "merchant_id"),
                               ("customer", custs, "customer_id"),
                               ("trigger", trigs, "id")]:
        for payload in source.values():
            status, _, ms = bot.push(scope, payload[key], payload)
            latencies["/v1/context"].append(ms)
            if status != 200:
                check(f"warmup push {scope}/{payload[key]}", False, f"HTTP {status}")
                break
    check("warmup accepted every base context",
          all(ok for label, ok in results if label.startswith("warmup push")) or
          not any(label.startswith("warmup push") for label, _ in results),
          f"{len(latencies['/v1/context'])} pushes")
    status, health, _ = bot.call("GET", "/v1/healthz")
    counts = health["contexts_loaded"]
    check("healthz counts match what was pushed",
          counts == {"category": len(cats), "merchant": len(merch),
                     "customer": len(custs), "trigger": len(trigs)}, str(counts))

    # --- tick, reply, suppression, adaptation -------------------------------
    status, body, ms = bot.tick([DIP])
    latencies["/v1/tick"].append(ms)
    actions = body.get("actions", [])
    check("tick returns one action", status == 200 and len(actions) == 1, f"{ms:.0f}ms")
    if not actions:
        return report()
    action = actions[0]
    check("action is wire-complete", set(action) == ACTION_FIELDS)
    check("action is specific and reasoned",
          "50%" in action["body"] and len(action["rationale"]) > 60,
          action["body"][:72])

    status, answer, ms = bot.reply(action["conversation_id"], "Yes, do it.", BHARAT)
    latencies["/v1/reply"].append(ms)
    check("reply moves to the action",
          status == 200 and answer["action"] == "send" and "CONFIRM" in answer["body"],
          f"{ms:.0f}ms")
    check("reply does not re-qualify",
          not any(p in answer["body"].lower() for p in
                  ("would you", "do you", "can you tell", "what if", "how about")))

    status, body, ms = bot.tick([DIP])
    latencies["/v1/tick"].append(ms)
    check("suppression holds the told story", body.get("actions") == [])

    deeper = copy.deepcopy(merch[BHARAT])
    deeper["performance"]["delta_7d"]["calls_pct"] = -0.37
    bot.push("merchant", BHARAT, deeper, 2)
    refreshed = copy.deepcopy(trigs[DIP])
    refreshed["payload"]["delta_pct"] = -0.37
    bot.push("trigger", DIP, refreshed, 2)
    status, body, ms = bot.tick([DIP])
    latencies["/v1/tick"].append(ms)
    fresh = body.get("actions", [])
    check("a refreshed context produces a fresh message",
          len(fresh) == 1 and "37%" in fresh[0]["body"] and "50%" not in fresh[0]["body"],
          fresh[0]["body"][:72] if fresh else "no action")

    status, stale, _ = bot.push("merchant", BHARAT, merch[BHARAT], 1)
    check("a lower version is refused",
          status == 409 and stale.get("current_version") == 2, stale.get("reason", ""))
    status, repeat, _ = bot.push("merchant", BHARAT, deeper, 2)
    check("an identical repeat is idempotent success",
          status == 200 and repeat.get("accepted") is True)

    # --- reply families -----------------------------------------------------
    families = [
        ("Yes", "send"), ("Yes, do it", "send"), ("Go ahead", "send"),
        ("No", "end"), ("Not interested", "end"),
        ("Not now", "wait"), ("Later", "wait"),
        ("How much is it?", "send"), ("Why?", "send"),
        ("Can you also help me with my GST filing?", "send"),
    ]
    for index, (message, expected) in enumerate(families):
        status, answer, ms = bot.reply(f"conv_smoke_{index}", message, MEERA)
        latencies["/v1/reply"].append(ms)
        ok = status == 200 and answer["action"] == expected
        if answer.get("action") == "send":
            ok = ok and bool(answer.get("body", "").strip()) and bool(answer.get("cta"))
        check(f"reply {message[:38]!r} -> {expected}", ok, answer.get("action", ""))

    canned = "Thanks for contacting us. We will get back to you shortly."
    sequence = [bot.reply("conv_smoke_auto", canned, AUTO_MERCHANT, turn_number=turn)[1]["action"]
                for turn in range(2, 7)]
    check("a repeated auto-reply terminates",
          sequence[0] == "send" and "end" in sequence and sequence.count("send") == 1,
          str(sequence))

    status, body, _ = bot.tick([RECALL])
    customer_action = (body.get("actions") or [None])[0]
    leaks = ("click-through", "percentage points", "peer", "median", "profile views")
    check("a customer send is separated from merchant analytics",
          customer_action is not None
          and customer_action["send_as"] == "merchant_on_behalf"
          and not any(term in customer_action["body"].lower() for term in leaks),
          customer_action["body"][:64] if customer_action else "no action")

    # --- latency ------------------------------------------------------------
    for path, samples in latencies.items():
        if not samples:
            continue
        ordered = sorted(samples)
        worst = ordered[-1]
        check(f"{path} within budget",
              worst < BUDGETS_MS[path],
              f"p50 {ordered[len(ordered) // 2]:.0f}ms max {worst:.0f}ms "
              f"(budget {BUDGETS_MS[path]}ms)")

    return report()


def report() -> int:
    failed = [label for label, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    for label in failed:
        print(f"  FAILED: {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
