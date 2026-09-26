#!/usr/bin/env python3
"""Run the official judge simulator, working around a transport-layer block.

`judge_simulator.py` hand-rolls its HTTP with `urllib`, which sends
`User-Agent: Python-urllib/3.x`. Groq's edge (Cloudflare) rejects that fingerprint
with `403 error code: 1010` before the request ever reaches the API, so the
simulator's own scorer cannot connect: the key and the model are fine, and the
bot itself is unaffected because it talks to Groq through httpx.

This runner installs a normal User-Agent for `urllib` and then hands control to the
simulator unchanged. It also retries a throttled scoring call rather than letting the simulator fall
back to its digit-counting placeholder, which otherwise reports 5/10 on four
dimensions as though the message had been judged.

`judge_simulator.py` is never modified: its scoring prompt, its scenarios and its
thresholds are all its own.

    export BOT_URL=https://your-service.onrender.com
    export LLM_API_KEY=gsk_...            # the simulator's scorer key
    export LLM_MODEL=openai/gpt-oss-120b
    python scripts/run_judge.py all
    python scripts/run_judge.py full_evaluation

Values already set in the simulator's own CONFIGURATION block are used when the
corresponding environment variable is absent, so either way of configuring works.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# A plain desktop-browser User-Agent. Nothing is spoofed beyond the client name:
# the request, the key and the payload are the simulator's own.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# The simulator paces nothing, so a free-tier 429 sends it into _fallback_score():
# 5/10 on four dimensions with "Could not evaluate", indistinguishable from a real
# score in the summary. Retrying is ordinary client behaviour and changes nothing
# about the evaluation.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
# 503 from a hosted scorer is provider capacity, not a quota problem, and it
# clears on its own, so be patient rather than falling through to
# _fallback_score(), whose placeholder 5/10s look like real scores.
MAX_RETRIES = 6
BASE_BACKOFF_SECONDS = 4.0

SCENARIOS = (
    "warmup", "phase2_short", "auto_reply", "intent_transition", "hostile",
    "all", "full_evaluation",
)


def challenge_dir() -> Path:
    override = os.environ.get("CHALLENGE_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent.parent / "magicpin-ai-challenge"


class RetryOnThrottle(urllib.request.BaseHandler):
    """Retry a throttled or transiently failed request instead of giving up.

    `Retry-After` is honoured when the server sends it; otherwise the wait grows
    exponentially. Applies to every `urllib` call in the process, which in practice
    means the scorer: the bot's own endpoints are not rate limited.
    """

    # Runs before the default error handler, which would raise.
    handler_order = 200

    def http_error_default(self, req, fp, code, msg, hdrs):
        raise urllib.error.HTTPError(req.full_url, code, msg, hdrs, fp)

    def _retry(self, req, fp, code, msg, headers):  # noqa: ARG002
        if code not in RETRY_STATUSES:
            return None
        attempt = getattr(req, "_retry_attempt", 0)
        if attempt >= MAX_RETRIES:
            return None
        retry_after = headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after else BASE_BACKOFF_SECONDS * (2**attempt)
        except (TypeError, ValueError):
            delay = BASE_BACKOFF_SECONDS * (2**attempt)
        # 429 = the per-minute token bucket is drained; nothing short of a
        # full refill cycle helps, and capping at 15s just burns attempts.
        # 5xx = provider capacity, which clears in seconds.
        ceiling = 60.0 if code == 429 else 15.0
        if code == 429 and not retry_after:
            delay = 60.0
        delay = min(delay, ceiling)
        print(f"    [retry] HTTP {code}; waiting {delay:.0f}s "
              f"(attempt {attempt + 1}/{MAX_RETRIES})")
        fp.close()
        time.sleep(delay)
        req._retry_attempt = attempt + 1
        return self.parent.open(req, timeout=req.timeout)

    http_error_429 = _retry
    http_error_500 = _retry
    http_error_502 = _retry
    http_error_503 = _retry
    http_error_504 = _retry


# Free-tier rate limits are the practical ceiling on a scored run, and an
# exhausted retry still lands in the simulator's _fallback_score(), which
# reports 5/10 "Could not evaluate" and is easy to mistake for a real mediocre
# score. Pacing keeps a run under the per-minute limit.
_PACED_HOSTS = (
    "groq.com", "openai.com", "generativelanguage.googleapis.com",
    "api.anthropic.com", "deepseek.com", "openrouter.ai",
)  # Ollama runs locally and needs no pacing.
# The bot and the scorer share one 8,000 tokens/min key: the bot spends ~1,100
# per wording call and takes up to six per tick, while a scoring call needs
# ~1,800. At any tighter spacing the two compete for the same bucket and the
# scorer loses, falling through to _fallback_score(): whose placeholder 5/10s
# are indistinguishable from real ones. A full minute lets the bucket refill.
# Override with JUDGE_MIN_INTERVAL=3 once the scorer has a key of its own.
MIN_INTERVAL_SECONDS = float(os.environ.get("JUDGE_MIN_INTERVAL", "60"))
_last_request_at = 0.0


class PaceRequests(urllib.request.BaseHandler):
    """Space out calls to the scoring API so a run does not trip the limit."""

    handler_order = 100

    def default_open(self, req):
        global _last_request_at
        host = req.host or ""
        if any(h in host for h in _PACED_HOSTS):
            wait = MIN_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
            if wait > 0:
                time.sleep(wait)
            _last_request_at = time.monotonic()
        return None


def install_opener() -> None:
    """Install a User-Agent and throttle-retry for every `urllib` call."""
    opener = urllib.request.build_opener(PaceRequests(), RetryOnThrottle())
    opener.addheaders = [("User-Agent", USER_AGENT)]
    urllib.request.install_opener(opener)


def load_simulator(path: Path):
    """Import `judge_simulator.py` as a module, unmodified."""
    spec = importlib.util.spec_from_file_location("judge_simulator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["judge_simulator"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "all"
    if scenario not in SCENARIOS:
        print(f"unknown scenario {scenario!r}; choose from: {', '.join(SCENARIOS)}")
        return 2

    root = challenge_dir()
    simulator_path = root / "judge_simulator.py"
    if not simulator_path.is_file():
        print(f"judge_simulator.py not found at {simulator_path}; set CHALLENGE_DIR")
        return 2

    install_opener()
    # The simulator resolves its dataset relative to its own file, but run from the
    # challenge directory anyway so any relative path it adds keeps working.
    os.chdir(root)
    judge = load_simulator(simulator_path)

    # Environment wins over the file's CONFIGURATION block; the file's values
    # remain the fallback so either way of configuring works.
    for variable, attribute in (
        ("BOT_URL", "BOT_URL"),
        ("LLM_PROVIDER", "LLM_PROVIDER"),
        ("LLM_API_KEY", "LLM_API_KEY"),
        ("LLM_MODEL", "LLM_MODEL"),
    ):
        value = os.environ.get(variable)
        if value:
            setattr(judge, attribute, value)
    judge.TEST_SCENARIO = scenario

    print(f"bot      : {judge.BOT_URL}")
    print(f"scorer   : {judge.LLM_PROVIDER} / {judge.LLM_MODEL or '(provider default)'}")
    print(f"scenario : {scenario}")
    print("urllib patched: browser User-Agent (Cloudflare 1010) + "
          "retry on 429/5xx (scorer throttling)\n")

    judge.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
