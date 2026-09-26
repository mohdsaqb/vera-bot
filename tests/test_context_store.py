"""ContextStore unit tests: version semantics, retrieval, reset."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from services.context_store import STALE_VERSION, ContextStore


def test_first_save_is_stored(fresh_store: ContextStore):
    result = fresh_store.save("merchant", "m_001", 1, {"views": 10})

    assert result.stored is True
    assert result.version == 1
    assert result.reason is None
    assert result.record.payload == {"views": 10}
    assert result.record.scope == "merchant"
    assert result.record.context_id == "m_001"


def test_higher_version_replaces(fresh_store: ContextStore):
    fresh_store.save("merchant", "m_001", 1, {"views": 10})
    result = fresh_store.save("merchant", "m_001", 2, {"views": 20})

    assert result.stored is True
    assert result.version == 2
    assert fresh_store.get_payload("merchant", "m_001") == {"views": 20}
    assert len(fresh_store) == 1


def test_same_version_is_a_noop(fresh_store: ContextStore):
    fresh_store.save("merchant", "m_001", 3, {"views": 10})
    result = fresh_store.save("merchant", "m_001", 3, {"views": 999})

    assert result.stored is False
    assert result.reason == STALE_VERSION
    assert result.version == 3
    assert fresh_store.get_payload("merchant", "m_001") == {"views": 10}


def test_lower_version_is_ignored(fresh_store: ContextStore):
    fresh_store.save("merchant", "m_001", 5, {"views": 50})
    result = fresh_store.save("merchant", "m_001", 4, {"views": 40})

    assert result.stored is False
    assert result.reason == STALE_VERSION
    assert result.version == 5
    assert fresh_store.get_payload("merchant", "m_001") == {"views": 50}


def test_rejected_save_reports_the_held_record(fresh_store: ContextStore):
    fresh_store.save("merchant", "m_001", 5, {"views": 50})
    result = fresh_store.save("merchant", "m_001", 2, {"views": 20})

    assert result.record.version == 5
    assert result.record.payload == {"views": 50}


def test_version_zero_is_accepted_then_superseded(fresh_store: ContextStore):
    assert fresh_store.save("merchant", "m_001", 0, {"n": 0}).stored is True
    assert fresh_store.save("merchant", "m_001", 0, {"n": 1}).stored is False
    assert fresh_store.save("merchant", "m_001", 1, {"n": 2}).stored is True
    assert fresh_store.get_payload("merchant", "m_001") == {"n": 2}


def test_large_version_jump_is_accepted(fresh_store: ContextStore):
    fresh_store.save("category", "dentists", 1, {"slug": "dentists"})
    assert fresh_store.save("category", "dentists", 99, {"slug": "dentists"}).stored


def test_is_stale_matches_save_behaviour(fresh_store: ContextStore):
    assert fresh_store.is_stale("merchant", "m_001", 1) is False  # unknown key

    fresh_store.save("merchant", "m_001", 3, {})

    assert fresh_store.is_stale("merchant", "m_001", 2) is True
    assert fresh_store.is_stale("merchant", "m_001", 3) is True
    assert fresh_store.is_stale("merchant", "m_001", 4) is False


def test_keys_are_scoped(fresh_store: ContextStore):
    fresh_store.save("merchant", "same_id", 1, {"kind": "merchant"})
    fresh_store.save("trigger", "same_id", 1, {"kind": "trigger"})
    fresh_store.save("customer", "same_id", 1, {"kind": "customer"})

    assert fresh_store.get_payload("merchant", "same_id") == {"kind": "merchant"}
    assert fresh_store.get_payload("trigger", "same_id") == {"kind": "trigger"}
    assert fresh_store.get_payload("customer", "same_id") == {"kind": "customer"}
    assert len(fresh_store) == 3


def test_replacing_one_key_leaves_others_untouched(fresh_store: ContextStore):
    fresh_store.save("merchant", "m_001", 1, {"n": 1})
    fresh_store.save("merchant", "m_002", 1, {"n": 2})
    fresh_store.save("merchant", "m_001", 2, {"n": 11})

    assert fresh_store.get_payload("merchant", "m_002") == {"n": 2}
    assert fresh_store.get_version("merchant", "m_002") == 1


def test_get_returns_none_for_unknown_key(fresh_store: ContextStore):
    assert fresh_store.get("merchant", "nope") is None
    assert fresh_store.get_payload("merchant", "nope") is None
    assert fresh_store.get_version("merchant", "nope") is None


def test_list_scope_returns_only_that_scope_sorted(fresh_store: ContextStore):
    fresh_store.save("merchant", "m_002", 1, {})
    fresh_store.save("merchant", "m_001", 1, {})
    fresh_store.save("trigger", "trg_001", 1, {})

    merchants = fresh_store.list_scope("merchant")

    assert [record.context_id for record in merchants] == ["m_001", "m_002"]
    assert fresh_store.list_scope("category") == []


def test_counts_are_zero_filled_for_all_scopes(fresh_store: ContextStore):
    assert fresh_store.counts() == {
        "category": 0, "merchant": 0, "customer": 0, "trigger": 0,
    }

    fresh_store.save("category", "dentists", 1, {})
    fresh_store.save("merchant", "m_001", 1, {})
    fresh_store.save("merchant", "m_002", 1, {})

    assert fresh_store.counts() == {
        "category": 1, "merchant": 2, "customer": 0, "trigger": 0,
    }


def test_clear_empties_the_store_and_reports_the_count(fresh_store: ContextStore):
    fresh_store.save("merchant", "m_001", 1, {})
    fresh_store.save("trigger", "trg_001", 1, {})

    assert fresh_store.clear() == 2
    assert len(fresh_store) == 0
    assert fresh_store.counts() == {
        "category": 0, "merchant": 0, "customer": 0, "trigger": 0,
    }
    assert fresh_store.clear() == 0


def test_stored_records_are_immutable(fresh_store: ContextStore):
    record = fresh_store.save("merchant", "m_001", 1, {}).record

    with pytest.raises(FrozenInstanceError):
        record.version = 2


def test_stored_at_is_timezone_aware(fresh_store: ContextStore):
    record = fresh_store.save("merchant", "m_001", 1, {}).record

    assert record.stored_at.tzinfo is not None
    assert record.stored_at.utcoffset().total_seconds() == 0


def test_concurrent_saves_keep_the_highest_version(fresh_store: ContextStore):
    from concurrent.futures import ThreadPoolExecutor

    versions = list(range(1, 101))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(
            lambda v: fresh_store.save("merchant", "m_001", v, {"v": v}), versions
        ))

    assert fresh_store.get_version("merchant", "m_001") == 100
    assert fresh_store.get_payload("merchant", "m_001") == {"v": 100}
    assert len(fresh_store) == 1
