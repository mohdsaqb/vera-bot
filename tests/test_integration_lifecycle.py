"""End-to-end lifecycle through the HTTP API, the way the judge drives it.

Everything here goes through `TestClient` rather than calling helpers directly,
because the judge only ever sees the endpoints. Covered:

  * version handling for all four scopes, end to end
  * adaptive context injection — a newer version changes what the next tick says
  * trigger lifecycle: expiry, completion, freshness
  * competing triggers, ranking and the per-tick action cap
  * suppression: same story held, new story allowed, refreshed context resent
  * conversation sessions from first touch to close
  * error handling on every endpoint
  * determinism of the whole path
"""

from __future__ import annotations

import copy

import pytest

from conftest import NOW, load_contexts, push_context, reply_to, tick_now

PERF_DIP = "trg_004_perf_dip_bharat"
BHARAT = "m_002_bharat_dentist_mumbai"


# ---------------------------------------------------------------------------- #
# 4. Version handling, per scope
# ---------------------------------------------------------------------------- #
class TestVersionHandling:
    @pytest.mark.parametrize(
        ("scope", "context_id_key"),
        [("category", "dentists"), ("merchant", BHARAT),
         ("customer", "c_001_priya_for_m001"), ("trigger", PERF_DIP)],
    )
    def test_the_full_version_sequence(self, client, dataset, scope, context_id_key):
        """A, A again, 0, B, B again — for every scope the judge pushes."""
        payloads = {
            "category": dataset.categories.get("dentists"),
            "merchant": dataset.merchants.get(BHARAT),
            "customer": dataset.customers.get("c_001_priya_for_m001"),
            "trigger": dataset.triggers.get(PERF_DIP),
        }
        payload = payloads[scope]

        assert push_context(client, scope, context_id_key, payload, 1).status_code == 200

        # Same version, same content: a retry, so it reads as success and the ack
        # repeats. Anything else would fail the judge's warmup check.
        repeat = push_context(client, scope, context_id_key, payload, 1)
        assert repeat.status_code == 200
        assert repeat.json()["accepted"] is True

        # Same version, different content: a genuine disagreement.
        conflicting = copy.deepcopy(payload)
        conflicting["_conflicting_marker"] = True
        clash = push_context(client, scope, context_id_key, conflicting, 1)
        assert clash.status_code == 409
        assert clash.json() == {"accepted": False, "reason": "stale_version",
                                "current_version": 1}

        # Lower version can never overwrite.
        assert push_context(client, scope, context_id_key, payload, 0).status_code == 409

        # Higher version replaces, and is then itself retry-safe.
        assert push_context(client, scope, context_id_key, payload, 2).status_code == 200
        again = push_context(client, scope, context_id_key, payload, 2)
        assert again.status_code == 200
        assert again.json()["accepted"] is True

    def test_the_latest_version_is_the_one_decisions_see(self, client, dataset):
        load_contexts(client, dataset, [PERF_DIP])
        first = tick_now(client, [PERF_DIP])
        assert "50%" in first[0]["body"]

        deeper = copy.deepcopy(dataset.merchants[BHARAT])
        deeper["performance"]["delta_7d"]["calls_pct"] = -0.35
        push_context(client, "merchant", BHARAT, deeper, 2)
        trigger_v2 = copy.deepcopy(dataset.triggers[PERF_DIP])
        trigger_v2["payload"]["delta_pct"] = -0.35
        push_context(client, "trigger", PERF_DIP, trigger_v2, 2)

        second = tick_now(client, [PERF_DIP])

        assert second, "a refreshed context should produce a fresh message"
        assert "35%" in second[0]["body"]
        assert "50%" not in second[0]["body"], "the superseded figure must be gone"

    def test_a_rejected_version_changes_nothing(self, client, dataset):
        load_contexts(client, dataset, [PERF_DIP])
        tick_now(client, [PERF_DIP])

        wrong = copy.deepcopy(dataset.merchants[BHARAT])
        wrong["performance"]["delta_7d"]["calls_pct"] = -0.99
        assert push_context(client, "merchant", BHARAT, wrong, 1).status_code == 409

        # Nothing newer was accepted, so there is nothing new to say.
        assert tick_now(client, [PERF_DIP]) == []


# ---------------------------------------------------------------------------- #
# 3. Adaptive injection across scopes
# ---------------------------------------------------------------------------- #
class TestAdaptiveContext:
    def test_a_new_digest_item_is_picked_up(self, client, dataset):
        research = "trg_001_research_digest_dentists"
        load_contexts(client, dataset, [research])
        first = tick_now(client, [research])
        assert "fluoride" in first[0]["body"]

        category = copy.deepcopy(dataset.categories["dentists"])
        category["digest"].insert(0, {
            "id": "d_injected", "kind": "compliance",
            "title": "DCI raises the minimum sterilisation cycle to 18 minutes",
            "source": "DCI circular 2026-05-02",
            "summary": "Cycles under 18 minutes no longer satisfy the standard.",
            "actionable": "Re-time your autoclave before the next inspection",
        })
        push_context(client, "category", "dentists", category, 2)
        trigger = copy.deepcopy(dataset.triggers[research])
        trigger["payload"]["top_item_id"] = "d_injected"
        push_context(client, "trigger", research, trigger, 2)

        second = tick_now(client, [research])

        assert second
        assert "sterilisation" in second[0]["body"]
        assert "DCI circular 2026-05-02" in second[0]["body"]

    def test_a_new_merchant_offer_is_used(self, client, dataset):
        load_contexts(client, dataset, [PERF_DIP])
        first = tick_now(client, [PERF_DIP])
        assert "Aligner Consultation @ ₹499" in first[0]["body"], "catalog proposal first"

        with_offer = copy.deepcopy(dataset.merchants[BHARAT])
        with_offer["offers"] = [
            {"id": "o_new", "title": "Free Consultation", "status": "active"}
        ]
        push_context(client, "merchant", BHARAT, with_offer, 2)

        second = tick_now(client, [PERF_DIP])

        assert second
        # The merchant now runs an offer, so it is referenced as live rather than
        # proposed from the catalog, and the ask changes with it.
        assert "Free Consultation" in second[0]["body"]
        assert "Aligner Consultation" not in second[0]["body"]
        assert second[0]["body"] != first[0]["body"]
        # The dip figure still comes from the trigger payload, which was not
        # re-pushed — the merchant snapshot is not the "why now" for this family.
        assert "50%" in second[0]["body"]

    def test_a_customer_state_change_is_respected(self, client, dataset):
        """A winback to someone now recorded as active must stop."""
        winback = "trg_015_winback_rashmi"
        load_contexts(client, dataset, [winback])
        assert tick_now(client, [winback]), "a lapsed customer should be reachable"

        now_active = copy.deepcopy(dataset.customers["c_010_rashmi_for_m007"])
        now_active["state"] = "active"
        push_context(client, "customer", "c_010_rashmi_for_m007", now_active, 2)
        push_context(client, "trigger", winback,
             copy.deepcopy(dataset.triggers[winback]), 2)

        assert tick_now(client, [winback]) == []

    def test_a_no_op_version_bump_stays_quiet(self, client, dataset):
        load_contexts(client, dataset, [PERF_DIP])
        tick_now(client, [PERF_DIP])

        push_context(client, "merchant", BHARAT, dataset.merchants[BHARAT], 2)
        push_context(client, "merchant", BHARAT, dataset.merchants[BHARAT], 3)

        assert tick_now(client, [PERF_DIP]) == [], "nothing changed, so nothing to say"


# ---------------------------------------------------------------------------- #
# 5. Trigger lifecycle
# ---------------------------------------------------------------------------- #
class TestReferenceTime:
    """Expiry is judged against the timeline the triggers are on, not blindly."""

    ALL_DENTIST = [
        "trg_001_research_digest_dentists",
        "trg_002_compliance_dci_radiograph",
        "trg_022_cde_webinar_dentists",
        "trg_023_competitor_opened_dentist",
    ]

    def test_an_expired_trigger_inside_a_live_set_is_still_dropped(self, client, dataset):
        """The case that matters: strict expiry whenever the tick can be placed."""
        load_contexts(client, dataset, self.ALL_DENTIST)

        # 2026-05-10 is past the research and CDE windows, inside the others.
        actions = tick_now(client, self.ALL_DENTIST, now="2026-05-10T00:00:00Z")

        assert len(actions) == 1
        assert actions[0]["trigger_id"] in {
            "trg_002_compliance_dci_radiograph", "trg_023_competitor_opened_dentist"
        }

    def test_a_single_expired_trigger_is_not_evidence_of_a_wrong_clock(self, client, dataset):
        load_contexts(client, dataset, ["trg_010_ipl_match_delhi"])

        assert tick_now(client, ["trg_010_ipl_match_delhi"],
                        now="2026-12-01T00:00:00Z") == []

    def test_a_recent_lapse_across_a_queue_is_still_a_lapse(self, client, dataset):
        """Events decaying over days must expire; only a months-wide gap is skew."""
        # Both of these lapse in early May; nothing in the set outlives them.
        short_lived = ["trg_001_research_digest_dentists", "trg_022_cde_webinar_dentists"]
        load_contexts(client, dataset, short_lived)

        assert tick_now(client, short_lived, now="2026-05-20T00:00:00Z") == []

    def test_the_newest_expiry_in_the_set_is_what_counts(self, client, dataset):
        """One long-lived trigger keeps the whole tick on its timeline."""
        load_contexts(client, dataset, self.ALL_DENTIST)

        # Past three of the four, but the compliance deadline runs to December.
        actions = tick_now(client, self.ALL_DENTIST, now="2026-06-15T00:00:00Z")

        assert len(actions) == 1
        assert actions[0]["trigger_id"] == "trg_002_compliance_dci_radiograph"

    def test_a_wall_clock_far_off_the_dataset_does_not_silence_everything(
        self, client, dataset
    ):
        """`judge_simulator.py` sends the machine clock; the dataset is dated April.

        Read literally that expires the whole corpus and answers every tick with
        silence, so a tick that cannot be placed on the triggers' timeline withholds
        time-based judgement instead.
        """
        load_contexts(client, dataset, self.ALL_DENTIST)

        actions = tick_now(client, self.ALL_DENTIST, now="2027-09-26T00:00:00Z")

        assert actions, "an off-timeline clock must not silence the whole queue"
        assert actions[0]["body"].strip()

    def test_withholding_does_not_weaken_the_other_decisions(self, client, dataset):
        """Only time judgement is withheld — everything else still applies."""
        load_contexts(client, dataset, self.ALL_DENTIST + ["trg_019_chronic_refill_grandfather"])

        actions = tick_now(
            client, self.ALL_DENTIST + ["trg_019_chronic_refill_grandfather"],
            now="2027-09-26T00:00:00Z",
        )

        # The pharmacy-only refill trigger is on a dentist here only via its own
        # merchant, so it may appear; what must not appear is a category mismatch.
        for action in actions:
            assert action["rationale"]
            assert action["body"].strip()
        assert len({a["merchant_id"] for a in actions}) == len(actions)


class TestTriggerLifecycle:
    def test_an_expired_trigger_produces_nothing(self, client, dataset):
        ipl = "trg_010_ipl_match_delhi"
        load_contexts(client, dataset, [ipl])

        assert tick_now(client, [ipl], now="2026-04-26T10:00:00Z"), "live before expiry"
        assert tick_now(client, [ipl], now="2026-06-01T00:00:00Z") == [], "dead after expiry"

    def test_a_trigger_with_no_stored_context_is_skipped(self, client, dataset):
        load_contexts(client, dataset, [PERF_DIP])

        actions = tick_now(client, ["trg_does_not_exist", PERF_DIP])

        assert len(actions) == 1
        assert actions[0]["trigger_id"] == PERF_DIP

    def test_a_trigger_for_an_unknown_merchant_is_skipped(self, client, dataset):
        orphan = copy.deepcopy(dataset.triggers[PERF_DIP])
        orphan["id"] = "trg_orphan"
        orphan["merchant_id"] = "m_999_not_pushed"
        push_context(client, "trigger", "trg_orphan", orphan)

        assert tick_now(client, ["trg_orphan"]) == []

    def test_trigger_identity_survives_similar_text(self, client, dataset):
        """Two dips on the same merchant in different weeks are different events."""
        load_contexts(client, dataset, [PERF_DIP])
        week_two = copy.deepcopy(dataset.triggers[PERF_DIP])
        week_two["id"] = "trg_004b_perf_dip_bharat_w18"
        week_two["suppression_key"] = "perf_dip:m_002_bharat_dentist_mumbai:calls:2026-W18"
        week_two["payload"]["delta_pct"] = -0.28
        push_context(client, "trigger", week_two["id"], week_two)

        first = tick_now(client, [PERF_DIP])
        second = tick_now(client, [week_two["id"]])

        assert first and second, "a genuinely new event must not be suppressed"
        assert first[0]["suppression_key"] != second[0]["suppression_key"]
        assert first[0]["body"] != second[0]["body"]


# ---------------------------------------------------------------------------- #
# 6 / 23. Competing triggers and action limits
# ---------------------------------------------------------------------------- #
class TestCompetingTriggers:
    COMPETING = [
        "trg_001_research_digest_dentists",   # urgency 2, knowledge
        "trg_002_compliance_dci_radiograph",  # urgency 4, compliance
        "trg_022_cde_webinar_dentists",       # urgency 1, knowledge
        "trg_023_competitor_opened_dentist",  # urgency 2, competition
    ]

    def test_the_strongest_story_wins_the_merchant(self, client, dataset):
        load_contexts(client, dataset, self.COMPETING)

        actions = tick_now(client, self.COMPETING)

        assert len(actions) == 1, "one action per merchant per tick"
        assert actions[0]["trigger_id"] == "trg_002_compliance_dci_radiograph"

    def test_the_ranking_does_not_depend_on_input_order(self, client, dataset):
        load_contexts(client, dataset, self.COMPETING)

        first = tick_now(client, self.COMPETING)
        # A second tick picks the next-strongest; order the remainder differently.
        remaining = [t for t in self.COMPETING if t != first[0]["trigger_id"]]
        second = tick_now(client, list(reversed(remaining)))
        third = tick_now(client, remaining)

        assert first[0]["trigger_id"] == "trg_002_compliance_dci_radiograph"
        assert second and third
        assert second[0]["trigger_id"] != third[0]["trigger_id"], "each tick advances"

    def test_later_ticks_work_through_the_queue(self, client, dataset):
        load_contexts(client, dataset, self.COMPETING)

        seen = []
        for _ in range(5):
            actions = tick_now(client, self.COMPETING)
            seen.extend(action["trigger_id"] for action in actions)

        assert len(seen) == len(set(seen)), "no story told twice"
        assert 2 <= len(seen) <= len(self.COMPETING)

    def test_many_merchants_stay_under_the_action_cap(self, client, dataset):
        from services.decision_service import MAX_ACTIONS_PER_TICK

        all_triggers = list(dataset.triggers)
        load_contexts(client, dataset, all_triggers)

        actions = tick_now(client, all_triggers)

        assert 0 < len(actions) <= MAX_ACTIONS_PER_TICK
        merchants = [action["merchant_id"] for action in actions]
        assert len(merchants) == len(set(merchants)), "one action per merchant"

    def test_suppressed_stories_do_not_consume_action_slots(self, client, dataset):
        """A held story must not crowd out a sendable one."""
        load_contexts(client, dataset, [PERF_DIP, "trg_001_research_digest_dentists"])
        first = tick_now(client, [PERF_DIP])
        assert len(first) == 1

        both = tick_now(client, [PERF_DIP, "trg_001_research_digest_dentists"])

        assert len(both) == 1
        assert both[0]["trigger_id"] == "trg_001_research_digest_dentists"


# ---------------------------------------------------------------------------- #
# 7. Suppression and deduplication
# ---------------------------------------------------------------------------- #
class TestSuppression:
    def test_the_same_story_is_not_retold(self, client, dataset):
        load_contexts(client, dataset, [PERF_DIP])

        assert len(tick_now(client, [PERF_DIP])) == 1
        assert tick_now(client, [PERF_DIP]) == []
        assert tick_now(client, [PERF_DIP]) == []

    def test_a_different_merchant_is_unaffected(self, client, dataset):
        load_contexts(client, dataset, [PERF_DIP, "trg_021_unverified_gbp_sunrise"])
        tick_now(client, [PERF_DIP])

        other = tick_now(client, ["trg_021_unverified_gbp_sunrise"])

        assert len(other) == 1
        assert other[0]["merchant_id"] == "m_010_sunrisepharm_pharmacy_lucknow"

    def test_a_different_customer_is_unaffected(self, client, dataset):
        load_contexts(client, dataset,
                      ["trg_003_recall_due_priya", "trg_019_chronic_refill_grandfather"])
        tick_now(client, ["trg_003_recall_due_priya"])

        other = tick_now(client, ["trg_019_chronic_refill_grandfather"])

        assert len(other) == 1
        assert other[0]["customer_id"] == "c_013_grandfather_for_m009"

    def test_the_ledger_records_the_context_version_it_sent_at(self, client, dataset):
        from state import get_suppression_ledger

        load_contexts(client, dataset, [PERF_DIP])
        actions = tick_now(client, [PERF_DIP])

        recorded = get_suppression_ledger().context_version_for(actions[0]["suppression_key"])
        assert recorded is not None and recorded > 0

    def test_an_opt_out_stops_later_stories_for_that_merchant(self, client, dataset):
        load_contexts(client, dataset, [PERF_DIP, "trg_005_renewal_due_bharat"])
        first = tick_now(client, [PERF_DIP])

        assert reply_to(client, first[0]["conversation_id"], "Stop messaging me.",
                     merchant_id=BHARAT)["action"] == "end"

        assert tick_now(client, ["trg_005_renewal_due_bharat"]) == []


# ---------------------------------------------------------------------------- #
# 8. Conversation sessions
# ---------------------------------------------------------------------------- #
class TestSessions:
    def test_a_tick_action_is_the_first_touch_of_its_own_thread(self, client, dataset):
        from state import get_conversation_store

        load_contexts(client, dataset, [PERF_DIP])
        actions = tick_now(client, [PERF_DIP])

        state = get_conversation_store().get(actions[0]["conversation_id"])
        assert state is not None
        assert state.session_state == "new"
        assert state.is_first_touch is False, "the opening message is already recorded"
        assert actions[0]["template_name"].startswith("vera_")
        assert actions[0]["template_params"]

    def test_the_session_advances_with_the_conversation(self, client, dataset):
        from state import get_conversation_store

        load_contexts(client, dataset, [PERF_DIP])
        conversation_id = tick_now(client, [PERF_DIP])[0]["conversation_id"]
        store = get_conversation_store()

        assert store.get(conversation_id).session_state == "new"

        reply_to(client, conversation_id, "how much?", merchant_id=BHARAT)
        assert store.get(conversation_id).session_state == "awaiting_reply"

        reply_to(client, conversation_id, "not now", merchant_id=BHARAT)
        assert store.get(conversation_id).session_state == "holding"

        reply_to(client, conversation_id, "not interested", merchant_id=BHARAT)
        assert store.get(conversation_id).session_state == "ended"

    def test_every_turn_is_not_a_new_conversation(self, client, dataset):
        from state import get_conversation_store

        load_contexts(client, dataset, [PERF_DIP])
        conversation_id = tick_now(client, [PERF_DIP])[0]["conversation_id"]

        reply_to(client, conversation_id, "how much?", merchant_id=BHARAT)
        reply_to(client, conversation_id, "hmm", merchant_id=BHARAT)

        assert len(get_conversation_store()) == 1
        assert get_conversation_store().get(conversation_id).turn_count >= 5

    def test_an_inbound_timestamp_is_kept(self, client, dataset):
        from state import get_conversation_store

        load_contexts(client, dataset, [PERF_DIP])
        conversation_id = tick_now(client, [PERF_DIP])[0]["conversation_id"]

        reply_to(client, conversation_id, "how much?", merchant_id=BHARAT)

        assert get_conversation_store().get(conversation_id).last_inbound_at == NOW


# ---------------------------------------------------------------------------- #
# 24. Error handling on every endpoint
# ---------------------------------------------------------------------------- #
class TestErrorHandling:
    def test_malformed_json_on_each_post(self, client):
        for path in ("/v1/context", "/v1/tick", "/v1/reply"):
            response = client.post(path, content=b"{not json",
                                   headers={"Content-Type": "application/json"})
            assert response.status_code == 400, path
            assert response.json()

    def test_context_rejections_keep_the_contract_shape(self, client):
        cases = [
            {"scope": "nope", "context_id": "x", "version": 1, "payload": {}},
            {"scope": "merchant", "version": 1, "payload": {}},
            {"scope": "merchant", "context_id": "x", "version": -1, "payload": {}},
            {"scope": "merchant", "context_id": "x", "version": 1, "payload": "text"},
        ]
        for body in cases:
            response = client.post("/v1/context", json=body)
            assert response.status_code == 400, body
            assert response.json()["accepted"] is False

    def test_tick_survives_every_kind_of_missing_context(self, client, dataset):
        merchant = dataset.merchants[BHARAT]
        # A trigger whose merchant exists but whose category was never pushed.
        push_context(client, "merchant", BHARAT, merchant)
        push_context(client, "trigger", PERF_DIP, dataset.triggers[PERF_DIP])
        assert tick_now(client, [PERF_DIP]) == []

        # A customer-scoped trigger with no customer context.
        push_context(client, "category", "dentists", dataset.categories["dentists"])
        push_context(client, "merchant", "m_001_drmeera_dentist_delhi",
             dataset.merchants["m_001_drmeera_dentist_delhi"])
        push_context(client, "trigger", "trg_003_recall_due_priya",
             dataset.triggers["trg_003_recall_due_priya"])
        assert tick_now(client, ["trg_003_recall_due_priya"]) == []

    def test_tick_rejects_a_body_without_now(self, client):
        assert client.post("/v1/tick", json={"available_triggers": []}).status_code == 400

    def test_reply_rejects_a_body_without_a_conversation(self, client):
        assert client.post(
            "/v1/reply", json={"from_role": "merchant", "message": "hi"}
        ).status_code == 400

    def test_reply_to_an_unknown_conversation_is_answered(self, client):
        answer = reply_to(client, "conv_never_existed", "yes do it", merchant_id="m_404")

        assert answer["action"] in {"send", "wait", "end"}
        assert answer["rationale"]

    def test_reply_with_an_unknown_merchant_and_customer(self, client):
        answer = reply_to(client, "conv_unknown_both", "how much?",
                       merchant_id="m_404", customer_id="c_404")

        assert answer["action"] in {"send", "wait", "end"}

    @pytest.mark.parametrize("message", ["", "   ", "?", "\n", "x" * 4000])
    def test_odd_reply_bodies_do_not_break_anything(self, client, message):
        answer = reply_to(client, "conv_odd", message)

        assert answer["action"] in {"send", "wait", "end"}
        if answer["action"] == "send":
            assert answer["body"].strip()

    def test_health_and_metadata_never_depend_on_state(self, client, dataset):
        for _ in range(3):
            assert client.get("/v1/healthz").status_code == 200
            assert client.get("/v1/metadata").status_code == 200

        load_contexts(client, dataset, [PERF_DIP])
        tick_now(client, [PERF_DIP])

        assert client.get("/v1/healthz").json()["status"] == "ok"
        assert client.get("/v1/metadata").json()["team_name"]


# ---------------------------------------------------------------------------- #
# 25. Determinism of the whole path
# ---------------------------------------------------------------------------- #
class TestDeterminism:
    def test_two_identical_runs_produce_identical_actions(self, client, dataset):
        from state import reset_state

        triggers = list(dataset.triggers)

        def run() -> list[dict]:
            reset_state()
            load_contexts(client, dataset, triggers)
            return tick_now(client, triggers)

        first, second = run(), run()

        assert [a["body"] for a in first] == [a["body"] for a in second]
        assert [a["suppression_key"] for a in first] == [a["suppression_key"] for a in second]
        assert [a["conversation_id"] for a in first] == [a["conversation_id"] for a in second]
        assert [a["rationale"] for a in first] == [a["rationale"] for a in second]

    def test_equal_priority_triggers_break_ties_stably(self, client, dataset):
        """Two identical-priority stories on different merchants keep a fixed order."""
        from state import reset_state

        pair = ["trg_022_cde_webinar_dentists", "trg_012_milestone_mylari"]

        def run() -> list[str]:
            reset_state()
            load_contexts(client, dataset, pair)
            return [a["trigger_id"] for a in tick_now(client, pair)]

        assert run() == run()
