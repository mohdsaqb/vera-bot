# Vera AI Challenge — merchant engagement bot

## Overview

Vera talks to local-commerce merchants on WhatsApp on magicpin's behalf: she
notices something worth telling them about, says it in one short message anchored
on a fact they can check, and asks for exactly one thing. She also writes to a
merchant's own customers, from the merchant's number, when the merchant's data
says it is warranted.

This is a rebuild of that product as an HTTP service the challenge's judge harness
drives. Given four context layers — the category, the merchant, the trigger that
fired, and optionally a customer — it decides whether to speak at all, what the
strongest verifiable reason is, which real offer applies, and the single next step
to ask for. Roughly half the triggers it is offered produce no message, each
refusal with a reason.

## Architecture

```
Category + Merchant + Trigger + Customer?
            │
            ▼
   Deterministic decision engine          engine/
   expiry · category fit · consent · contradiction checks
   trigger ranking · signal selection · offer selection
   action + CTA + sender + suppression key
            │
            ▼
      MessagePlan  ──projection──►  GenerationBrief
            │                             │
            │                             ▼
            │                   LLM wording (Llama via Groq)
            │                             │
            │                      structured {body, cta}
            │                             │
            ▼                             ▼
   deterministic body  ◄──fallback── grounding validator
            │
            ▼
   /v1/tick action  ·  /v1/reply answer
```

The engine decides. The model only rewrites the body. If the model is missing,
slow, broken, or produces a fact it was not given, the engine's own wording goes
out instead and nothing upstream changes.

## Tech stack

| Purpose | Choice |
|---|---|
| HTTP | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Configuration | python-dotenv + environment variables |
| Wording | LangChain (`langchain-core`, `langchain-groq`) → Llama on Groq |
| State | in-process Python dicts behind locks |
| Tests | pytest |

Nothing else. No database, no queue, no vector store, no agent framework, no
Docker. The wording packages are imported lazily, so the service starts and
serves correctly even if they are absent.

## Determinism

Every business decision is made in Python and is reproducible: the same inputs
always produce the same trigger choice, the same signal, the same offer, the same
action, the same CTA, the same sender, the same suppression key and the same
rationale. No randomness anywhere; no clock is read unless the caller passes
`now`.

The model is confined to wording. It never sees the trigger queue, the priority
scores, the provenance paths or the suppression key — only a `GenerationBrief`
holding the facts the decision selected. Its output is checked against that brief
by containment (every figure, amount, date and proper noun must appear in it) and
discarded on any mismatch. `tests/test_llm_integration.py` asserts field by field
that a generated message and a deterministic one differ only in `body`.

## State

In-process and deliberately so: four singletons (contexts, conversations, the
suppression ledger, the writer) behind `threading.RLock`, reset by
`POST /v1/teardown`.

The judge pushes context over hundreds of calls and expects it remembered for the
test window; the privacy rule forbids keeping any of it afterwards. State
therefore survives requests but **not a restart or redeploy**. That is acceptable
because the harness keeps one instance alive for a slot — but it means a redeploy
mid-slot loses everything, so do not deploy while a test is running.

## API

| Endpoint | Purpose |
|---|---|
| `GET /v1/healthz` | liveness and per-scope context counts |
| `GET /v1/metadata` | bot identity |
| `POST /v1/context` | versioned ingestion of the four context scopes |
| `POST /v1/tick` | proactive sends for this moment |
| `POST /v1/reply` | answer an inbound merchant or customer turn |

Plus the optional `POST /v1/teardown` from the testing brief, which wipes state.

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

```bash
curl http://127.0.0.1:8000/v1/healthz
curl http://127.0.0.1:8000/v1/metadata
```

> `--host 0.0.0.0` binds IPv4 only. On macOS prefer `127.0.0.1` over `localhost`
> in local requests, since `localhost` may resolve to IPv6 `::1` first.

To have Llama word the messages, set the environment (see `.env.example`):

```bash
export LLM_PROVIDER=groq
export LLM_MODEL=llama-3.3-70b-versatile   # must be one your account serves
export GROQ_API_KEY=...                    # never committed
```

Model ids are retired over time, so check
[console.groq.com/docs/models](https://console.groq.com/docs/models) first. A
wrong id, a bad key or an unreachable provider degrades to deterministic wording
rather than failing a request. The startup log says which layer is active:

```
INFO vera.api wording layer: llm (provider=groq model=llama-3.3-70b-versatile)
INFO vera.api wording layer: deterministic (provider=none reason=llm_provider_not_configured)
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q                 # 658 tests
```

No test makes a network call; the model is always mocked. Tests read the real
challenge files from `../magicpin-ai-challenge` (override with `CHALLENGE_DIR`)
and skip cleanly when it is absent — 294 of the 658 need no dataset at all. The
canonical-pairs suites additionally need the generated dataset:

```bash
cd ../magicpin-ai-challenge
python3 dataset/generate_dataset.py --seed-dir dataset --out expanded
```

### Smoke-testing a running instance

`scripts/smoke_test.py` drives a live server through the whole judge-shaped flow —
warmup, tick, reply, suppression, a version bump, the reply families, audience
separation and per-endpoint latency — and exits non-zero if anything fails, so it
can gate a deploy:

```bash
python scripts/smoke_test.py                              # local
python scripts/smoke_test.py https://vera.onrender.com    # a deployment
```

It calls `/v1/teardown` first, so never point it at an instance in the middle of a
judged slot. Until `TEAM_NAME` and `CONTACT_EMAIL` are set it will fail exactly one
check, *team identity is not a placeholder* — that is the reminder, not a bug.

## Judge simulator

`judge_simulator.py` ships with the challenge. Edit its configuration block:

```python
BOT_URL = "https://<your-deployment>"   # or http://127.0.0.1:8000
LLM_PROVIDER = "groq"                   # or openai, anthropic, gemini, ...
LLM_API_KEY = "<your key>"              # the simulator's own scorer key
LLM_MODEL = ""                          # optional
TEST_SCENARIO = "all"
```

```bash
cd ../magicpin-ai-challenge && python judge_simulator.py
```

Its warmup pushes every base context at version 1 and treats any
`accepted: false` as a failed warmup, so **start from clean state**: restart the
process, or `curl -X POST <base>/v1/teardown` first. Against a server whose
versions have already advanced, a version-1 push is correctly refused and the
warmup reports a failure that is not one.

## Deployment

Render, via the checked-in `render.yaml` blueprint — no Dockerfile, no build
image to maintain:

```
GitHub repository → Render Blueprint → Uvicorn → https://<service>.onrender.com
```

1. Push this directory to a GitHub repository.
2. Render → **New → Blueprint** → select the repository. `render.yaml` supplies
   the runtime, build command, start command and health check path.
3. Set the secret and identity variables in the dashboard (they are marked
   `sync: false` so they are never read from the repository):
   `GROQ_API_KEY`, `LLM_PROVIDER`, `LLM_MODEL`, `TEAM_NAME`, `TEAM_MEMBERS`,
   `CONTACT_EMAIL`.
4. Verify the deployment end to end, then submit the URL:

   ```bash
   curl https://<service>.onrender.com/v1/healthz
   curl https://<service>.onrender.com/v1/metadata
   python scripts/smoke_test.py https://<service>.onrender.com
   ```

   The judge calls `https://<service>.onrender.com/v1/healthz` and the rest under
   `/v1/*`.

A `Procfile` is included for Railway, Heroku and anything else that reads one.
The start command is identical in every environment:

```
uvicorn app:app --host 0.0.0.0 --port $PORT
```

### Deployment notes

- **Free tiers idle out.** Render's free plan sleeps an inactive service and the
  first request then pays a cold start of tens of seconds. The judge disqualifies
  a bot after three consecutive `healthz` failures, so use a paid instance for a
  real slot, or keep the service warm.
- **Do not redeploy during a test slot.** State is in-process; a redeploy restarts
  it and everything pushed so far is lost.
- **One instance only.** State is per-process, so a second replica would answer
  with a different view of the world. Keep the instance count at 1.
- **Confirm the Python version** in the first build log. `.python-version` and
  `render.yaml` both request 3.13; local verification was on 3.14.6.

## The decision engine

```
compose(category, merchant, trigger, customer=None, now=None) -> ComposedMessage
```

```
 normalize ──► evaluate trigger ──► select primary signal ──► supporting fact
                     │                                              │
                     │ (hard blocks)                                ▼
                     ▼                                        select offer
              no_action + reason                                    │
                                                                    ▼
   body ◄── render ◄── MessagePlan ◄── suppression key ◄── select action + CTA
     │                                                       + sender identity
     ▼
  validate ──► ComposedMessage (send | no_action)
```

Stage by stage:

1. **Normalize** (`engine/normalize.py`) — the four payloads become frozen
   dataclasses. Field names were read out of the dataset, not guessed, and every
   optional block may be absent: the 40 generated merchants carry no offers,
   signals, review themes or history.
2. **Evaluate the trigger** (`engine/triggers.py`) — first the hard blocks, each
   describing a message that must not exist: expired, meaningless for this
   vertical, contradicted by the merchant's own numbers, no consent for this
   purpose, no verifiable fact, a dated event too far off to prepare for. What
   survives is scored as a weighted sum of urgency, merchant relevance,
   actionable value, category fit and time pressure — weights and the reasoning
   behind them are in `engine/policy.py`. `rank_triggers` orders a queue, ties
   breaking on urgency then trigger id so the order never depends on input order.
3. **Select the primary signal** (`engine/signals.py`) — the single strongest
   verifiable fact, which is the "why now". The trigger's own payload wins when
   it has content; for knowledge and compliance families the digest item it
   points at wins, carrying its citation. When the payload is one of the 75
   generated placeholders, a merchant-derived fact stands in — unless the family
   *is* its payload (an appointment with no time, a competitor with no name),
   in which case the engine declines.
4. **Select one supporting fact** — and only one. A research item that names a
   patient segment is tied to the merchant's own count for it; a recall is tied
   to the cohort it touches; a local chapter event is tied to the merchant's
   city. A customer-facing message may only use facts about that customer.
5. **Select the offer** (`engine/offers.py`) — the merchant's own active offer
   first, then a category catalog pattern marked as a proposal, then nothing.
   Nothing is ever synthesised, expired offers are never offered, and an offer
   irrelevant to the moment is dropped rather than bolted on.
6. **Select the action and CTA** (`engine/actions.py`) — exactly one ask, from
   the challenge's CTA taxonomy, naming its object. Sender identity follows the
   trigger scope. The suppression key is the trigger's own when it supplies one,
   otherwise synthesised from family, recipient, action and time bucket.
7. **MessagePlan** (`engine/types.py`) — the fact-only hand-off object. Every
   fact carries the dotted path it came from, so a generator consuming the plan
   cannot claim a number it was not given. This is the seam Phase 3 plugs into.
8. **Render and validate** (`engine/render.py`) — category voice from
   `voice.salutation_examples` and `voice.tone`, Hindi-English code-mix for
   Hindi-speaking recipients, the ask last. The body is then checked for URLs
   (a hard fail per the judge's table), taboo vocabulary from
   `voice.vocab_taboo`, unresolved placeholders and more than one question. A
   body that fails downgrades to no-action rather than going out.

### Restraint is a decision

Over the full 100-trigger expanded set the engine sends 44 and declines 56, each
with a reason code: `trigger_claim_not_supported_by_data`,
`trigger_not_relevant_to_category`, `no_verifiable_fact_available`,
`customer_consent_missing_for_purpose`, `customer_opted_out_of_reminders`,
`customer_not_contactable`, `event_too_distant_to_act_on`, `trigger_expired`,
`not_worth_interrupting`, `required_context_missing`. Two examples of why:

* a `perf_dip` trigger whose merchant's numbers are *rising* is declined — the
  merchant can check that claim and find it false;
* a `chronic_refill_due` trigger assigned to a dentist is declined — the
  expanded dataset assigns generated triggers at random, and a prescription
  refill does not exist in that vertical.

## The wording layer

Optional, and off by default. With `LLM_PROVIDER` unset the deterministic
renderer writes every message and the process makes no network calls at all.

```
MessagePlan ──► GenerationBrief ──► prompt ──► Llama (Groq, via LangChain)
                     │                              │
                     │                       structured output {body, cta}
                     │                              │
                     │                              ▼
                     └──────────────────► grounding validator
                                                    │
                                    grounded? ──────┴────── not grounded?
                                        │                        │
                                   use the body          use the deterministic body
```

### What the model is and is not allowed to do

It receives a `GenerationBrief`: the category and its voice, the recipient's
name, the facts the decision selected, the chosen offer and whether it is live,
the single ask, the CTA label, the language, and the deterministic rendering for
reference. That is the whole input — it is a projection of the plan
(`engine/brief.py`), so the model never sees the trigger queue, the priority
scores, the provenance paths, the suppression key or any context the decision
did not select.

It returns `{body, cta}`. The `cta` is requested only so the model has to
acknowledge the decided call to action; the returned value is discarded. The body
is the single useful output, and even that is provisional.

Everything else comes out of `compose()` exactly as it went in: which trigger
fired, which signal anchors the message, which offer is used, the action, the
CTA, the sender identity, the suppression key, the conversation id, the
rationale, and whether to send at all. `test_llm_integration.py` asserts field by
field that a generated message and a deterministic one differ only in `body`.

### Grounding

`services/message_validator.py` checks a generated body against its brief by
containment, not by meaning — a figure is acceptable only if that exact figure
appears in the brief. Blunt on purpose: it cannot be talked around.

| Check | Catches |
|---|---|
| numbers, ₹ amounts, percentages | invented metrics, prices, discounts |
| dates | invented deadlines and appointment times |
| proper nouns | invented competitors, brands, offer names, places |
| offer identity | the chosen offer silently swapped or dropped |
| offer state | a suggested offer described as already running |
| snake_case tokens, system vocabulary | internal identifiers, "as an AI", prompts |
| one question, one request, one action verb per ask | a second CTA |
| category `vocab_taboo` | "guaranteed", "best in city", medical overclaims |
| URLs | a hard fail in the judge's penalty table |
| audience | merchant analytics shown to a customer |

Two normalisations keep the check from rejecting faithful rewordings: a
possessive is folded to its stem (`JIDA's` → `JIDA`) and month names to three
letters (`October` → `Oct`). Everything that survives the engine's own renderer
passes this validator — 44 outbound bodies and 220 reply drafts across the
expanded dataset, verified.

### Failing safely

Fallback to the deterministic body on: no provider configured, no API key, the
packages absent, a client that will not build, a timeout, any provider
exception, a malformed or empty structured response, any validation failure, and
a per-tick budget running out. Nothing in this path raises to the caller —
`/v1/tick` and `/v1/reply` cannot fail because a model did.

Bounded work, since `/v1/tick` has a 10-second budget and may carry 20 actions:
one call per message, no repair round-trips, no second model grading the first, a
per-call timeout (`LLM_TIMEOUT_SECONDS`, default 6 s) and a per-tick ceiling of
`LLM_MAX_CALLS_PER_TICK` calls inside `LLM_TICK_BUDGET_SECONDS`. Messages past
the ceiling render deterministically.

The API key is read from the environment at the moment a client is built. It is
not on `Settings`, so there is no path from it to `/v1/metadata`; provider errors
are logged by exception type only, never as objects that might echo a request
header.

## Conversations and replies

`/v1/tick` opens a conversation for every action it returns, so the reply that
comes back lands on a thread that knows what it was about. State is in-memory and
per-process (`services/conversation_store.py`): the turns both ways, the turn
count, the last intent, whether the thread is active, consecutive auto-replies,
unanswered nudges, and the brief the thread was built on.

`engine/intent.py` classifies each inbound message deterministically, in a fixed
precedence — a keyword alone is not enough, because the same words mean different
things in company:

| Intent | Recognised by | What happens |
|---|---|---|
| `auto_reply` | canned business phrasing, or a verbatim repeat | flag once → back off 24 h → close |
| `hostile` | abuse, "stop messaging me", "unsubscribe" | close, and suppress the merchant |
| `reject` | "no", "not interested", "skip it" | close without re-pitching |
| `delay` | "not now", "later", "tomorrow" | back off 4 h or 24 h, thread stays open |
| `off_topic` | GST, tax, loans, payroll, licences | decline in one clause, return to the ask |
| `question` | interrogative shape or wording | answer from the stored facts, or say it is not known |
| `action_request` | "activate it", "send it now" | move to the action |
| `accept` | leading affirmation, "let's do it" | move to the action |
| `ambiguous` | none of the above | one plain restatement → back off → close |

Three behaviours are worth calling out, because each is a documented failure of
production Vera:

* **"Yes, do it. What's next?" is an acceptance, not a question.** A commitment
  that asks how to proceed is checked ahead of precedence, so the follow-through
  half cannot reclassify the whole message. The reply then names the next step and
  asks for one confirmation; it never asks another qualifying question, and that
  is enforced rather than assumed (`re_qualifies`).
* **Nothing is ever claimed as done.** There is no execution tool behind this
  bot, so an acceptance produces "I'll set X up and send it here to approve before
  it goes live. Reply CONFIRM and I'll proceed." — the next supported step, not a
  false completion.
* **Auto-replies are counted per merchant, not just per thread.** The same phone
  answers every conversation, and `judge_simulator.py` rotates the conversation id
  on each of its four canned turns, so per-thread counting would never notice.

## Layout

```
vera-bot/
├── app.py                        # FastAPI app + the 6 route handlers (thin)
├── config.py                     # env-backed Settings (no secrets, cached)
├── models.py                     # Pydantic request/response models = the contract
├── state.py                      # process-wide singletons + reset for tests
├── engine/                       # decisions — no I/O, no clock, no model calls
│   ├── types.py                  #   frozen dataclasses: MessagePlan, brief, state
│   ├── policy.py                 #   every tunable rule, weight and threshold
│   ├── normalize.py              #   raw JSON -> normalized contexts
│   ├── signals.py                #   fact extraction + primary/supporting choice
│   ├── triggers.py               #   evaluate_trigger, rank_triggers, consent
│   ├── offers.py                 #   select_offer
│   ├── actions.py                #   action, CTA, sender, suppression key
│   ├── intent.py                 #   classify_reply — deterministic, no model
│   ├── reply.py                  #   decide_reply: send / wait / end + the draft
│   ├── brief.py                  #   what a writer is allowed to see
│   ├── fmt.py                    #   how numbers and dates are written
│   ├── render.py                 #   deterministic body + output validation
│   └── compose.py                #   compose(), build_plan(), write_body()
├── services/
│   ├── context_store.py          # versioned in-memory store (storage only)
│   ├── suppression.py            # what has been said, when, at which version
│   ├── conversation_store.py     # threads in flight
│   ├── llm_service.py            # the only module that talks to a provider
│   ├── message_validator.py      # grounding checks on generated text
│   ├── decision_service.py       # store <-> engine bridge for /v1/tick
│   └── reply_service.py          # store <-> engine bridge for /v1/reply
├── tests/                        # 231 tests
│   ├── test_health.py            #   /v1/healthz
│   ├── test_metadata.py          #   /v1/metadata, incl. no secret leakage
│   ├── test_context.py           #   /v1/context over HTTP + /v1/teardown
│   ├── test_context_store.py     #   store version semantics
│   ├── test_tick_reply.py        #   endpoint contract floors
│   ├── test_normalize.py         #   normalization against the real schema
│   ├── test_signals.py           #   materiality, wording, fact choice
│   ├── test_triggers.py          #   eligibility, consent, ranking
│   ├── test_offers.py            #   offer preference and no fabrication
│   ├── test_compose.py           #   the A-O scenario matrix
│   ├── test_canonical_pairs.py   #   the 30 canonical (merchant, trigger) pairs
│   ├── test_tick_decisions.py    #   decision service + sending discipline
│   ├── test_intent.py            #   reply intent classification
│   ├── test_reply_intents.py     #   /v1/reply behaviour + conversation state
│   ├── test_message_validator.py #   grounding checks
│   ├── test_llm_service.py       #   the writer, with the model mocked
│   ├── test_llm_integration.py   #   writer in the pipeline + determinism
│   ├── test_integration_lifecycle.py  # versions, adaptive context, suppression
│   ├── test_integration_replay.py     # replay scenarios over HTTP
│   ├── test_canonical_endpoints.py    # the 30 pairs through /v1/tick
│   └── test_production_hardening.py   # error handler, access log, portability
├── scripts/
│   └── smoke_test.py             # drive a live deployment through the contract
├── conftest.py                   # TestClient, state reset, dataset fixtures, HTTP drivers
├── pytest.ini
├── requirements.txt              # runtime only — what the deployment installs
├── requirements-dev.txt          # adds pytest + httpx
├── render.yaml                   # Render blueprint: build, start, health check
├── Procfile                      # same start command for Railway/Heroku
├── .python-version               # 3.13
├── .env.example                  # every variable the code reads, no secrets
└── .gitignore
```

Responsibilities are split so each layer can be reasoned about alone:

- **`models.py`** owns the wire contract. `Scope`/`SCOPES` are defined once and
  derived from each other, so the validator and the healthz counts cannot drift.
- **`services/context_store.py`** owns *storage only* — version comparison,
  retrieval, per-scope counts. No decision logic.
- **`services/decision_service.py`** owns resolution and per-tick discipline:
  one action per merchant per tick, at most 20 in total, nothing already sent
  under the same suppression key, no repeated body in a conversation.
- **`engine/`** owns the decisions and holds no state, performs no I/O and reads
  no clock — `now` is a parameter, absent by default. It depends on a
  `MessageWriter` protocol, never on a provider SDK.
- **`services/llm_service.py`** is the only module that imports a provider. It
  can fail in every way and the worst outcome is deterministic wording.
- **`app.py`** handlers only translate between HTTP and those layers.

## Assumptions

1. **Same version → `409`.** The testing brief calls a same-version re-push
   "a no-op", and `api-call-examples.md` §1.5 shows exactly `409` with
   `{"accepted": false, "reason": "stale_version", "current_version": 1}` for it.
   Both hold here: nothing is overwritten *and* the response is the `409` from the
   example. Lower versions are rejected identically.
2. **`ack_id` format** is not specified beyond the examples (`ack_dentists_v1`,
   `ack_trg_001_v1`), so it is `ack_{context_id}_v{version}` — deterministic, and
   the same for a repeated push of the same version.
3. **Timestamps in, string; timestamps out, `...123Z`.** `delivered_at`, `now` and
   `received_at` are accepted as opaque strings rather than parsed, so an
   unexpected but valid ISO-8601 variant can never reject an otherwise-good
   request. Timestamps we emit (`stored_at`, `submitted_at`) use the judge's
   millisecond-plus-`Z` format.
4. **Permissive where the contract is silent, strict where it is explicit.**
   `scope` is validated against the four documented values because the contract
   defines an `invalid_scope` rejection for it; `from_role`, `turn_number` and
   `merchant_id`/`customer_id` on `/v1/reply` are permissive because
   `api-call-examples.md` §2.5 omits some of them. Unknown extra envelope keys
   are tolerated so a future judge field cannot 400 a valid push.
5. **`/v1/reply` returns `wait`, not `send`.** With no conversation logic yet, an
   empty `send` body would be scored as malformed (`-2`) and a canned body would
   risk the anti-repetition penalty. `wait` is contract-valid and costs nothing.
6. **Counts are not preloaded from the dataset.** `contexts_loaded` must read all
   zeros before warmup (§1.1), so nothing is loaded at startup and no challenge
   data is hardcoded.
7. **`version: 0` is accepted** as a first version (the contract never states a
   minimum); only negative versions are rejected.

Decision-engine judgement calls, each a one-line change in `engine/policy.py`:

8. **A trigger the merchant's data contradicts is declined, not softened.** The
   dataset assigns generated triggers to merchants at random, so a `perf_dip` can
   land on a merchant whose numbers rose. Asserting a fall the merchant can check
   and disprove is worse than silence.
9. **Consent is checked per purpose.** The seed customers pair each trigger with
   the matching scope (Priya's recall with `recall_reminders`, Mr. Sharma's refill
   with `refill_reminders`), which reads as deliberate. Broad
   `promotional_offers` consent authorises promotional outreach but not a
   clinical or medication reminder.
10. **`customer.state` gates customer families.** A winback aimed at someone
    recorded as `active` is the trigger being wrong about the person. Dates are
    used for the wording, the state label for eligibility — the generated
    customers all share the same visit dates, so the label is the real signal.
11. **A dated event needs a 45-day runway** (`MAX_EVENT_LEAD_DAYS`) unless its
    payload names an open preparation window. Diwali 188 days out has no "why
    now"; Kavya's bridal trigger says `next_step_window_open`, so it does.
12. **A 15% week-on-week move on a meaningful base is the materiality floor**
    (`MIN_MATERIAL_DELTA`, `MIN_METRIC_BASE`). +5% of 14 calls is under one call.
13. **Regional language mixes render in English.** `hi`/`hi-en mix` recipients
    get Hindi-English code-mix; `ta-en`, `te-en` and `kn-en` get English rather
    than Hindi, which would be the wrong language for that recipient.
14. **The trigger's own `suppression_key` is echoed when supplied.** The judge
    generates it to identify the event and the API examples echo it back on the
    action; re-deriving it would risk two keys for one story.

Integration decisions:

15. **An identical re-push of a held version reports success, not conflict.** The
    brief calls a repeat "idempotent" and the examples show a `409` for one, but
    the warmup check treats any `accepted: false` as a failed warmup that
    disqualifies the bot for that slot. A retry of identical content leaves the
    store exactly as the caller asked, so it answers `200`; a same-version push
    carrying *different* content is a real disagreement and still conflicts.
16. **A told story may be retold when its context has been refreshed.** The judge
    injects updated performance snapshots and new digest items mid-test and scores
    whether later sends reflect them. So suppression releases a key only when a
    newer context version has arrived *and* the message it produces has changed —
    a version bump that changes nothing stays quiet, and so does a reworded body
    with no new facts.
17. **Expiry is judged against the triggers' own timeline.** `judge_simulator.py`
    sends the machine's wall clock, which sits months past a dataset dated April
    2026 — read literally that expires 96 of 100 triggers and answers every tick
    with silence. When a tick's `now` is more than 30 days past the newest expiry
    among *several* available triggers, the clock is treated as off-timeline and
    time-based judgement is withheld for that tick. Any tick that can be placed
    keeps strict per-trigger expiry, which is the case that matters.

Wording-layer and reply decisions:

18. **Grounding is containment, not judgement.** A figure is allowed only if that
    exact figure is in the brief. This rejects some faithful rewordings — a model
    that converts "18 calls" to "fewer than twenty" loses the number and falls
    back — and that trade is deliberate: a false negative costs one deterministic
    message, a false positive costs a fabricated claim.
19. **The CTA the model returns is discarded.** It is requested so the model must
    acknowledge the decided call to action, and the deterministic value is the one
    that ships.
20. **A hostile reply or an explicit no closes the merchant, not just the
    thread.** Later triggers for that merchant are suppressed, because an opt-out
    is about the channel rather than the topic.
21. **An acceptance never claims completion.** There is no execution tool, so the
    reply states the next supported step and asks for one confirmation.
22. **Auto-reply counting is per merchant as well as per thread**, because one
    phone answers every conversation.
23. **`/v1/reply` is answered synchronously from stored state.** No model call is
    made for a `wait` or an `end`, and a question is answered from the facts the
    thread already holds — never from a fresh lookup or a guess.

---

## Roadmap

The API surface, the decision engine, the wording layer and reply handling are
all in place. What is not built:

- **No execution.** Nothing actually activates an offer, publishes a post or
  books a slot; every action is expressed as the next step pending confirmation.
  A real integration is the next thing that would change the product.
- **No cadence planning.** Each tick decides independently. Optimal sequencing
  inside a 24-hour session window (open challenge #3) is not modelled.
- **No first-touch template gating.** `template_name` and `template_params` are
  emitted for every action, but the 24-hour session rule is not tracked, so the
  bot does not know whether a given send needs the template or may be free-form.

Known weak spots, in order of how much they would move a score:

1. **A category-wide regulation notice has no merchant-specific second fact.**
   `regulation_change` anchors on the digest item and its `actionable` line; there
   is nothing in the merchant context that makes a dose-limit revision specific to
   one practice.
2. **Generated customers are thin.** They carry no `services_received` and share
   one visit date, so a winback to one reaches ~150 characters where a seed
   customer reaches ~300.
3. **Grounding cannot see paraphrase.** Rewordings that restate a number in words,
   or convert a unit, fall back. Measured on scripted rewordings, roughly one in
   five faithful rewrites is rejected this way.
4. **Reply drafts are templated per intent.** Within an intent, two threads read
   alike in structure. The model smooths this when configured, but the fallback
   does not.
5. **Intent classification is keyword-based.** It handles the documented cases and
   the replay scenarios, but a reply that is sarcastic, mixed-intent beyond the
   follow-through case, or in Devanagari script rather than transliteration will
   land in `ambiguous` — which backs off rather than guessing, but does not help.
6. **No real-provider test in CI.** Every test mocks the model. The live path was
   verified by hand against Groq (a real call, rejected on the key), so the client
   construction and structured-output setup are known to work, but no automated
   test covers a successful real completion.

## Verified locally

- `pytest -q` — **658 passed**, no network access in any test. 294 of them pass
  with no challenge dataset present, so the service is portable on its own.
- **Clean-environment install**: a fresh venv, `pip install -r requirements.txt`
  (41 packages), started with `PORT=10000 uvicorn app:app --host 0.0.0.0 --port
  $PORT`, confirmed listening on `*:10000` rather than loopback.
- **35-point production smoke test** against that clean deployment: the full
  context → tick → reply → suppression → version-bump flow, eleven reply
  phrasings, repeated auto-replies terminating, customer/merchant separation, and
  per-endpoint latency against the contract's budgets.
- **A 35-point acceptance checklist**, passing both with no provider configured
  and with one configured but unreachable.
- **The official `judge_simulator.py`**, run unmodified against the live service:
  warmup, auto-reply, intent-transition and hostile scenarios all PASS. Its LLM
  scorer was stubbed (no credentials available here), so the numeric scores carry
  no signal — the four conversation scenarios do not use the scorer and are real.
- **Latency** (deterministic path): healthz and metadata under 1 ms, a context
  push ~1 ms, a tick offering all 100 triggers 19 ms against a 10-second budget.
  With a configured-but-dead provider in the loop the worst tick was 385 ms.
- **Grounding self-check**: all 44 outbound bodies and 220 reply drafts the engine
  produces pass the validator that guards generated text.
- **Security**: no `.env`, no credential files, no key-shaped literals; the API key
  is read in exactly one function and is absent from `Settings`, `/v1/metadata`,
  `/v1/healthz`, prompts and logs — each asserted by a test.
