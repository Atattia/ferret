"""Regression test for macOS main-thread-only window creation."""

from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer

from main import SearchToggleDispatcher


class _SearchBar:
    def __init__(self):
        self.calls = []

    def isVisible(self):
        self.calls.append(("visible", threading.get_ident()))
        return False

    def hide(self):
        self.calls.append(("hide", threading.get_ident()))

    def show_and_focus(self):
        self.calls.append(("show", threading.get_ident()))


class MainThreadDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def test_worker_signal_runs_search_toggle_on_qt_thread(self):
        main_thread = threading.get_ident()
        search_bar = _SearchBar()
        dispatcher = SearchToggleDispatcher(search_bar)

        worker = threading.Thread(target=dispatcher.toggle_requested.emit)
        worker.start()
        worker.join()

        loop = QEventLoop()
        QTimer.singleShot(20, loop.quit)
        loop.exec()

        self.assertEqual([name for name, _thread in search_bar.calls], ["visible", "show"])
        self.assertTrue(all(thread == main_thread for _name, thread in search_bar.calls))


if __name__ == "__main__":
    unittest.main()
