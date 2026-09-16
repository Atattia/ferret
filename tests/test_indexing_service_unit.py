"""Dependency-free unit tests for the single-writer indexing coordinator."""

from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.indexing_service import IndexingService, JobKind


class IndexingServiceTests(unittest.TestCase):
    def make_service(self, **overrides):
        defaults = {
            "indexer": lambda path: None,
            "deleter": lambda path: None,
            "mover": lambda source, destination: None,
            "retry_delay": 0,
        }
        defaults.update(overrides)
        return IndexingService(**defaults)

    def test_latest_file_intent_coalesces_before_start(self):
        calls = []
        service = self.make_service(
            indexer=lambda path: calls.append(("index", path)),
            deleter=lambda path: calls.append(("delete", path)),
        )

        self.assertTrue(service.submit_index("document.txt"))
        self.assertFalse(service.submit_index("document.txt"))
        self.assertFalse(service.submit_delete("document.txt"))
        self.assertEqual(service.status.pending, 1)

        self.assertTrue(service.start())
        self.assertFalse(service.start())
        self.assertTrue(service.wait_for_idle(1))
        self.assertTrue(service.stop())
        self.assertTrue(service.stop())

        self.assertEqual(calls, [("delete", str(Path("document.txt").resolve()))])
        self.assertEqual(service.status.submitted, 1)
        self.assertEqual(service.status.coalesced, 2)
        self.assertEqual(service.status.completed, 1)

    def test_all_write_jobs_run_sequentially_on_one_worker(self):
        calls = []
        worker_threads = set()

        def record(kind):
            def call(path):
                calls.append((kind, Path(path).name))
                worker_threads.add(threading.get_ident())

            return call

        service = self.make_service(
            indexer=record("index"), deleter=record("delete")
        )
        service.submit_index("one.txt")
        service.submit_delete("two.txt")
        service.submit_index("three.md")

        service.start()
        self.assertTrue(service.wait_for_idle(1))
        service.stop()

        self.assertEqual(
            calls,
            [("index", "one.txt"), ("delete", "two.txt"), ("index", "three.md")],
        )
        self.assertEqual(len(worker_threads), 1)
        self.assertNotIn(threading.get_ident(), worker_threads)

    def test_enqueue_move_runs_as_one_atomic_coordinator_job(self):
        calls = []
        service = self.make_service(
            mover=lambda source, destination: calls.append(
                ("move", Path(source).name, Path(destination).name)
            ),
        )

        self.assertTrue(service.enqueue_move("old.txt", "new.txt"))
        service.start()
        self.assertTrue(service.wait_until_idle(1))
        service.stop()

        self.assertEqual(calls, [("move", "old.txt", "new.txt")])

    def test_default_move_indexes_destination_when_source_was_not_indexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            indexed = []
            service = IndexingService(Path(tmp) / "ferret.db", retry_delay=0)
            with (
                patch("core.indexer.move_indexed_file", return_value=False),
                patch(
                    "core.indexer.index_file",
                    side_effect=lambda path, *_args, **_kwargs: indexed.append(path),
                ),
            ):
                service.enqueue_move("old.txt", "new.txt")
                service.start()
                self.assertTrue(service.wait_for_idle(1))
                service.stop()

            self.assertEqual(len(indexed), 1)
            self.assertTrue(indexed[0].endswith("new.txt"))

    def test_transient_failures_retry_then_succeed(self):
        attempts = []

        def flaky(path):
            attempts.append(path)
            if len(attempts) < 3:
                raise OSError("temporarily unavailable")

        service = self.make_service(indexer=flaky, max_retries=2)
        service.submit_index("eventually.txt")
        service.start()
        self.assertTrue(service.wait_for_idle(1))
        service.stop()

        self.assertEqual(len(attempts), 3)
        self.assertEqual(service.status.retried, 2)
        self.assertEqual(service.status.completed, 1)
        self.assertEqual(service.status.failed, 0)
        self.assertEqual(service.failures, ())

    def test_permanent_failure_is_reported_without_stopping_queue(self):
        indexed = []

        def index(path):
            if path.endswith("bad.txt"):
                raise ValueError("bad document")
            indexed.append(Path(path).name)

        service = self.make_service(indexer=index)
        service.submit_index("bad.txt")
        service.submit_index("good.txt")
        service.start()
        self.assertTrue(service.wait_for_idle(1))
        service.stop()

        self.assertEqual(indexed, ["good.txt"])
        self.assertEqual(service.status.failed, 1)
        self.assertEqual(service.status.completed, 1)
        failure = service.status.last_failure
        self.assertIsNotNone(failure)
        self.assertEqual(failure.job.kind, JobKind.INDEX)
        self.assertEqual(failure.error_type, "ValueError")
        self.assertEqual(failure.message, "bad document")
        self.assertEqual(failure.attempts, 1)

    def test_scan_filters_candidates_and_reports_discovery_progress(self):
        indexed = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = [
                root / "one.txt",
                root / "two.PDF",
                root / "image.png",
                root / "ignored" / "three.md",
            ]
            service = self.make_service(
                indexer=lambda path: indexed.append(Path(path).name),
                scanner=lambda folder: candidates,
                exclude_patterns=["ignored"],
            )
            self.assertTrue(service.submit_scan(root))
            self.assertFalse(service.submit_scan(root))
            service.start()
            self.assertTrue(service.wait_for_idle(1))
            service.stop()

        self.assertEqual(indexed, ["one.txt", "two.PDF"])
        self.assertEqual(service.status.discovered, 2)
        # The scan itself and its two generated file jobs completed.
        self.assertEqual(service.status.completed, 3)
        self.assertEqual(service.status.coalesced, 1)

    def test_cancel_drops_pending_jobs_but_allows_active_call_to_finish(self):
        entered = threading.Event()
        release = threading.Event()
        indexed = []

        def blocking_index(path):
            indexed.append(Path(path).name)
            entered.set()
            release.wait(1)

        service = self.make_service(indexer=blocking_index)
        service.submit_index("active.txt")
        service.start()
        self.assertTrue(entered.wait(1))
        service.submit_index("cancel-one.txt")
        service.submit_index("cancel-two.txt")

        try:
            self.assertFalse(service.cancel(timeout=0.01))
            self.assertEqual(service.status.pending, 0)
            self.assertEqual(service.status.cancelled, 2)
        finally:
            release.set()
            self.assertTrue(service.stop(drain=False, timeout=1))

        self.assertEqual(indexed, ["active.txt"])
        self.assertEqual(service.status.completed, 1)
        self.assertFalse(service.status.running)

    def test_drain_processes_pending_jobs_and_service_can_restart(self):
        indexed = []
        service = self.make_service(
            indexer=lambda path: indexed.append(Path(path).name)
        )
        service.submit_index("first.txt")
        service.start()
        self.assertTrue(service.stop(drain=True, timeout=1))
        service.submit_index("second.txt")
        self.assertTrue(service.start())
        self.assertTrue(service.stop(drain=True, timeout=1))

        self.assertEqual(indexed, ["first.txt", "second.txt"])
        self.assertEqual(service.status.completed, 2)


if __name__ == "__main__":
    unittest.main()
