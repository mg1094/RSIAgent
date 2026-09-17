"""Memory ownership rules.

These are the invariants that make "the knowledge actually retained" mean
something.  Each test names the sentence in the paper it pins.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rsiagent.errors import MemoryIntegrityError
from rsiagent.memory.bank import MemoryBank, compute_stats


class TestStats:
    def test_deterministic_hash(self, tmp_path: Path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        for root in (a, b):
            (root / "notes").mkdir(parents=True)
            (root / "notes" / "x.md").write_text("hello")
        assert compute_stats(a).tree_hash == compute_stats(b).tree_hash

    def test_content_change_changes_hash(self, tmp_path: Path):
        root = tmp_path / "m"
        root.mkdir()
        (root / "x.md").write_text("one")
        first = compute_stats(root).tree_hash
        (root / "x.md").write_text("two")
        assert compute_stats(root).tree_hash != first

    def test_empty_bank(self, tmp_path: Path):
        stats = compute_stats(tmp_path / "missing")
        assert stats.file_count == 0
        assert stats.total_bytes == 0


class TestSessions:
    def test_edits_are_invisible_until_commit(self, tmp_path: Path):
        """Appendix A.2: work-phase edits are discarded before learning."""
        bank = MemoryBank(tmp_path / "memory")
        session = bank.session()
        session.write("draft.md", "scratch")
        assert bank.is_empty, "a session edit must not reach canonical memory"

        session.commit()
        assert not bank.is_empty
        assert bank.files()["draft.md"] == b"scratch"

    def test_discard_promotes_nothing(self, tmp_path: Path):
        bank = MemoryBank(tmp_path / "memory")
        session = bank.session()
        session.write("draft.md", "scratch")
        session.discard()
        assert bank.is_empty

    def test_closed_session_rejects_further_edits(self, tmp_path: Path):
        bank = MemoryBank(tmp_path / "memory")
        session = bank.session()
        session.commit()
        with pytest.raises(MemoryIntegrityError):
            session.write("more.md", "x")

    def test_delete_removes_a_file(self, tmp_path: Path):
        bank = MemoryBank(tmp_path / "memory")
        first = bank.session()
        first.write("a.md", "1")
        first.write("b.md", "2")
        first.commit()

        second = bank.session()
        second.delete("a.md")
        second.commit()
        assert set(bank.files()) == {"b.md"}

    def test_frozen_bank_refuses_writes(self, tmp_path: Path):
        """Phase 3: 'the curriculum agent and all memory updates are disabled'."""
        bank = MemoryBank(tmp_path / "memory", writeback=False)
        with pytest.raises(MemoryIntegrityError):
            bank.session().write("a.md", "x")


class TestSnapshots:
    def test_snapshot_is_isolated_from_later_commits(self, tmp_path: Path):
        """A wave's branches all read the same pre-wave state."""
        bank = MemoryBank(tmp_path / "memory")
        seed = bank.session()
        seed.write("base.md", "original")
        seed.commit()

        snapshot = bank.snapshot(label="wave0")

        later = bank.session()
        later.write("base.md", "changed")
        later.write("new.md", "added")
        later.commit()

        assert snapshot.read_all() == {"base.md": "original"}

    def test_two_sessions_on_one_snapshot_cannot_see_each_other(self, tmp_path: Path):
        """'Projects in the same wave ... cannot see or depend on sibling work.'"""
        bank = MemoryBank(tmp_path / "memory")
        snapshot = bank.snapshot(label="wave0")

        a = bank.session(snapshot)
        b = bank.session(snapshot)
        a.write("a.md", "from a")

        assert "a.md" not in b.read_all()
        assert "a.md" not in snapshot.read_all()

    def test_sequential_commits_see_preceding_ones(self, tmp_path: Path):
        """'Each update operates on the latest canonical memory, including
        preceding updates from the same wave.'"""
        bank = MemoryBank(tmp_path / "memory")
        pre_wave = bank.snapshot(label="wave0")

        first = bank.session(pre_wave)
        first.write("first.md", "1")
        first.commit()

        second = bank.session()  # opened on live canonical, not the snapshot
        assert "first.md" in second.read_all()

    def test_readonly_view_has_no_commit(self, tmp_path: Path):
        """'The curriculum agent may inspect a disposable copy ... its local
        edits are not synchronized back.'"""
        bank = MemoryBank(tmp_path / "memory")
        view = bank.readonly_view()
        assert not hasattr(view, "commit")
        assert not hasattr(view, "write")


class TestFreezing:
    def test_freeze_records_and_verifies_a_hash(self, tmp_path: Path):
        bank = MemoryBank(tmp_path / "memory")
        session = bank.session()
        session.write("a.md", "1")
        session.commit()

        frozen = bank.freeze_to(tmp_path / "frozen")
        bank.assert_frozen(frozen)  # unchanged: no exception

    def test_tampering_with_frozen_memory_is_detected(self, tmp_path: Path):
        bank = MemoryBank(tmp_path / "memory")
        session = bank.session()
        session.write("a.md", "1")
        session.commit()

        frozen = bank.freeze_to(tmp_path / "frozen")
        (frozen.path / "a.md").write_text("tampered")
        with pytest.raises(MemoryIntegrityError):
            bank.assert_frozen(frozen)

    def test_promotion_is_atomic_and_replaces(self, tmp_path: Path):
        bank = MemoryBank(tmp_path / "memory")
        session = bank.session()
        session.write("keep.md", "1")
        session.write("drop.md", "2")
        session.commit()

        replacement = bank.session()
        replacement.delete("drop.md")
        replacement.commit()

        assert set(bank.files()) == {"keep.md"}

    def test_render_reports_an_empty_bank(self, tmp_path: Path):
        assert MemoryBank(tmp_path / "memory").render() == "(memory is empty)"
