"""Offscreen checks for stale results and bounded worker lifetime."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel
from ui.searchbar import SearchBar, ResultItemWidget

# Establish the widget-capable singleton during discovery, before tests that
# only need QCoreApplication create the less capable base class.
_APP = QApplication.instance() or QApplication([])


class SearchBarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.bar = SearchBar("/unused.db")
        self.addCleanup(self.cleanup_bar)

    def cleanup_bar(self):
        self.bar._worker = None
        self.bar._shutdown()
        self.bar.close()
        self.bar.deleteLater()

    def test_clear_invalidates_inflight_results(self):
        self.bar._on_text_changed("old")
        generation = self.bar._generation
        self.bar._on_text_changed("")
        self.bar._on_results([], generation)
        self.assertEqual(self.bar.height(), 66)
        self.assertTrue(self.bar._status.isHidden())
        self.assertEqual(self.bar._pending_query, "")

    def test_typing_invalidates_results_before_debounce(self):
        self.bar._on_text_changed("old")
        generation = self.bar._generation
        self.bar._on_text_changed("new")
        self.bar._on_results([], generation)
        self.assertEqual(self.bar._status.text(), "Searching…")

    def test_worker_finishes_then_starts_only_latest_query(self):
        old = MagicMock()
        old.generation = 0
        self.bar._worker = old
        self.bar._on_text_changed("intermediate")
        self.bar._on_text_changed("latest")
        self.bar._debounce.stop()
        with patch("ui.searchbar.SearchWorker") as constructor:
            self.bar._run_search()
            constructor.assert_not_called()
            self.bar._worker_finished()
            constructor.assert_called_once_with("latest", "/unused.db", self.bar.model_path, self.bar._generation)
            constructor.return_value.start.assert_called_once()
        old.deleteLater.assert_called_once()

    def test_result_text_is_plain_and_includes_evidence(self):
        widget = ResultItemWidget("<b>name.pdf", "/docs/name.pdf", "<img src=x>", 0, 1,
                                  result={"matched_by": ["fts", "semantic"], "page": 4})
        labels = widget.findChildren(QLabel)
        content = [label for label in labels if label.text().startswith("<")]
        self.assertTrue(all(label.textFormat() == Qt.TextFormat.PlainText for label in content))
        self.assertTrue(any("Keyword · Meaning · Page 4" in label.text() for label in labels))
        widget.deleteLater()


if __name__ == "__main__":
    unittest.main()
