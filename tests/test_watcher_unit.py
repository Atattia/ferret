"""Focused lifecycle tests for the filesystem watcher.

These tests deliberately mock watchdog's observer.  They verify path
configuration and lifecycle behavior without starting a real background thread
or indexing any documents.
"""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import watcher


class _Event:
    def __init__(self, src_path, dest_path=None, is_directory=False):
        self.src_path = str(src_path)
        self.dest_path = str(dest_path) if dest_path is not None else None
        self.is_directory = is_directory


def _observer():
    observer = MagicMock()
    observer.is_alive.return_value = False
    observer.schedule.side_effect = lambda handler, path, recursive: (path, recursive)
    return observer


class FolderWatcherTests(unittest.TestCase):
    def test_event_handler_only_queues_supported_mutations(self):
        service = MagicMock()
        handler = watcher.FerretEventHandler(service, exclude_patterns=[".git"])
        handler.on_created(_Event("/docs/new.txt"))
        handler.on_modified(_Event("/docs/new.txt"))
        handler.on_deleted(_Event("/docs/new.txt"))
        handler.on_moved(_Event("/docs/old.txt", "/docs/new-name.md"))
        handler.on_created(_Event("/docs/.git/ignored.md"))
        handler.on_created(_Event("/docs/image.png"))

        self.assertEqual(service.enqueue_index.call_count, 2)
        service.enqueue_delete.assert_called_once()
        service.enqueue_move.assert_called_once()

    def test_missing_folder_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            observer = _observer()
            with patch.object(watcher, "Observer", return_value=observer):
                folder_watcher = watcher.FolderWatcher(
                    "/tmp/ferret-test.db", indexing_service=MagicMock()
                )
                self.assertFalse(folder_watcher.add_folder(Path(tmp) / "missing"))
                self.assertEqual(folder_watcher.watched_folders, set())
                folder_watcher.start()

            observer.start.assert_not_called()
            observer.schedule.assert_not_called()

    def test_reconfigure_adds_and_unschedules_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            first.mkdir()
            second.mkdir()
            observer = _observer()

            with patch.object(watcher, "Observer", return_value=observer):
                folder_watcher = watcher.FolderWatcher(
                    "/tmp/ferret-test.db", indexing_service=MagicMock()
                )
                folder_watcher.start()
                folder_watcher.reconfigure([first, second, first])
                folder_watcher.start()
                self.assertEqual(
                    folder_watcher.watched_folders,
                    {str(first.resolve()), str(second.resolve())},
                )
                folder_watcher.reconfigure([second])

            self.assertEqual(observer.schedule.call_count, 2)
            observer.unschedule.assert_called_once_with((str(first.resolve()), True))
            self.assertEqual(folder_watcher.watched_folders, {str(second.resolve())})

    def test_stop_is_idempotent_and_restart_recreates_observer(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "documents"
            folder.mkdir()
            first_observer = _observer()
            second_observer = _observer()

            with patch.object(
                watcher, "Observer", side_effect=[first_observer, second_observer]
            ):
                folder_watcher = watcher.FolderWatcher(
                    "/tmp/ferret-test.db", indexing_service=MagicMock()
                )
                folder_watcher.add_folder(folder)
                folder_watcher.start()
                folder_watcher.stop()
                folder_watcher.stop()
                folder_watcher.start()

            first_observer.stop.assert_called_once_with()
            first_observer.join.assert_called_once_with()
            second_observer.schedule.assert_called_once()
            second_observer.start.assert_called_once_with()

    def test_reconfigure_while_running_schedules_new_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            first.mkdir()
            second.mkdir()
            observer = _observer()

            with patch.object(watcher, "Observer", return_value=observer):
                folder_watcher = watcher.FolderWatcher(
                    "/tmp/ferret-test.db", indexing_service=MagicMock()
                )
                folder_watcher.add_folder(first)
                folder_watcher.start()
                folder_watcher.reconfigure([first, second])

            self.assertEqual(observer.schedule.call_count, 2)
            self.assertEqual(
                folder_watcher.watched_folders,
                {str(first.resolve()), str(second.resolve())},
            )


if __name__ == "__main__":
    unittest.main()
