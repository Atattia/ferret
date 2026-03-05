import subprocess
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

SCORE_CUTOFF = 0.45
FONT = "'Helvetica Neue', 'Ubuntu', 'Noto Sans', sans-serif"
W = 680          # fixed window width

_INPUT_H  = 66   # height of just the input row
_SEP_H    = 1    # hairline separator
_ROW_H    = 62   # height per result row
_PAD_H    = 8    # top+bottom padding inside the results list
_STATUS_H = 46   # "Searching…" / "No results" row
_MAX_ROWS = 7    # cap visible rows before scrolling

# Apple system colors
EXT_ICON = {
    ".pdf":  ("#FF453A", "PDF"),   # red
    ".docx": ("#0A84FF", "DOC"),   # blue
    ".doc":  ("#0A84FF", "DOC"),
    ".txt":  ("#8E8E93", "TXT"),   # gray
    ".md":   ("#FF9F0A",  "MD"),   # amber
}
_DEFAULT_ICON = ("#5E5CE6", "FILE")  # purple


def _row_alpha(rank: int, total: int) -> int:
    """Title text alpha (0-255): full at rank-0, fades to ~130 for the last."""
    if total <= 1:
        return 232
    return int(232 - (rank / (total - 1)) * 102)


class ResultItemWidget(QWidget):
    def __init__(self, filename: str, path: str, snippet: str,
                 rank: int, total: int, parent=None):
        super().__init__(parent)
        self.path = path

        ext = Path(filename).suffix.lower()
        icon_color, icon_text = EXT_ICON.get(ext, _DEFAULT_ICON)
        ta = _row_alpha(rank, total)
        sa = max(80, ta - 100)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 18, 0)
        layout.setSpacing(14)

        # Rounded-square file-type icon (mimics macOS app icon shape)
        icon = QLabel(icon_text)
        icon.setFixedSize(40, 40)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(f"""
            background: {icon_color};
            color: white;
            border-radius: 10px;
            font-size: 11px;
            font-weight: bold;
            font-family: monospace;
        """)
        layout.addWidget(icon, 0, Qt.AlignmentFlag.AlignVCenter)

        # Text block
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        name = QLabel(filename)
        name.setStyleSheet(
            f"color: rgba(255,255,255,{ta});"
            f"font-size: {'14px' if rank == 0 else '13px'};"
            f"font-weight: {'500' if rank == 0 else '400'};"
            f"font-family: {FONT};"
        )

        sub_text = (snippet[:95].replace("\n", " ").strip()
                    if snippet else str(Path(path).parent))
        sub = QLabel(sub_text)
        sub.setStyleSheet(
            f"color: rgba(255,255,255,{sa});"
            f"font-size: 12px; font-weight: 400;"
            f"font-family: {FONT};"
        )

        col.addWidget(name)
        col.addWidget(sub)
        layout.addLayout(col, 1)


class SearchWorker(QThread):
    results_ready = pyqtSignal(list, int)

    def __init__(self, query: str, db_path: str, model_path: str, generation: int):
        super().__init__()
        self.query = query
        self.db_path = db_path
        self.model_path = model_path
        self.generation = generation

    def run(self):
        try:
            from core.searcher import search
            results = search(self.query, self.db_path, top_k=8, model_path=self.model_path)
        except Exception as e:
            print(f"[searchbar] Search error: {e}")
            results = []
        self.results_ready.emit(results, self.generation)


class SearchBar(QWidget):
    def __init__(self, db_path: str, model_path: str = "~/ferret/models/bge-small-en"):
        super().__init__()
        self.db_path = db_path
        self.model_path = model_path
        self._worker: SearchWorker | None = None
        self._generation = 0
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(350)
        self._debounce.timeout.connect(self._run_search)
        self._pending_query = ""
        self._setup_ui()
        self._setup_hotkeys()

    def _setup_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(W)
        self.setFixedHeight(_INPUT_H)

        self._container = QFrame(self)
        self._container.setObjectName("container")
        self._container.setFixedWidth(W)
        self._container.setFixedHeight(_INPUT_H)
        self._container.setStyleSheet("""
            QFrame#container {
                background: rgba(26, 26, 28, 0.96);
                border: 0.5px solid rgba(255,255,255,0.13);
                border-radius: 20px;
            }
        """)

        col = QVBoxLayout(self._container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        # ── Input row ──────────────────────────────────────────────────────
        input_area = QWidget()
        input_area.setFixedHeight(_INPUT_H)
        input_area.setStyleSheet("background: transparent;")

        row = QHBoxLayout(input_area)
        row.setContentsMargins(18, 0, 18, 0)
        row.setSpacing(12)

        mag = QLabel("⌕")
        mag.setStyleSheet(f"color: rgba(255,255,255,0.35); font-size: 24px;")
        row.addWidget(mag, 0, Qt.AlignmentFlag.AlignVCenter)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Search your files…")
        self._input.setStyleSheet(f"""
            QLineEdit {{
                background: transparent;
                color: rgba(255,255,255,0.92);
                border: none;
                font-size: 18px;
                font-weight: 300;
                font-family: {FONT};
                padding: 0; margin: 0;
            }}
        """)
        self._input.textChanged.connect(self._on_text_changed)
        row.addWidget(self._input, 1)

        esc_lbl = QLabel("esc")
        esc_lbl.setStyleSheet(
            f"color: rgba(255,255,255,0.18); font-size: 11px; font-family: {FONT};"
        )
        row.addWidget(esc_lbl, 0, Qt.AlignmentFlag.AlignVCenter)

        col.addWidget(input_area)

        # ── Hairline separator ─────────────────────────────────────────────
        self._sep = QFrame()
        self._sep.setFrameShape(QFrame.Shape.HLine)
        self._sep.setFixedHeight(_SEP_H)
        self._sep.setStyleSheet("border: none; background: rgba(255,255,255,0.1);")
        self._sep.hide()
        col.addWidget(self._sep)

        # ── Results ────────────────────────────────────────────────────────
        self._results_list = QListWidget()
        self._results_list.setStyleSheet("""
            QListWidget {
                background: transparent;
                border: none;
                outline: none;
                padding: 4px 0;
            }
            QListWidget::item {
                padding: 0; margin: 0;
            }
            QListWidget::item:selected {
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
            }
            QListWidget::item:hover:!selected {
                background: rgba(255,255,255,0.05);
                border-radius: 8px;
            }
        """)
        self._results_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._results_list.hide()
        self._results_list.itemActivated.connect(self._open_result)
        col.addWidget(self._results_list)

        # ── Status label ───────────────────────────────────────────────────
        self._status = QLabel("")
        self._status.setFixedHeight(_STATUS_H)
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet(
            f"color: rgba(255,255,255,0.28); font-size: 13px; font-family: {FONT};"
        )
        self._status.hide()
        col.addWidget(self._status)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._container)

        self._center_on_screen()

    def _set_height(self, extra: int):
        """Grow/shrink window from the input downward. Never moves the top edge."""
        h = _INPUT_H + extra
        self._container.setFixedHeight(h)
        self.setFixedHeight(h)

    def _setup_hotkeys(self):
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.hide)

    def _center_on_screen(self):
        screen = QApplication.primaryScreen().geometry()
        self.move(
            screen.center().x() - W // 2,
            screen.center().y() - _INPUT_H // 2,   # center on just the input
        )

    def _on_text_changed(self, text: str):
        text = text.strip()
        if not text:
            self._debounce.stop()
            self._sep.hide()
            self._results_list.hide()
            self._status.hide()
            self._set_height(0)
            return

        self._pending_query = text
        self._sep.show()
        self._results_list.hide()
        self._status.setText("Searching…")
        self._status.show()
        self._set_height(_SEP_H + _STATUS_H)
        self._debounce.start()

    def _run_search(self):
        query = self._pending_query
        if not query:
            return
        self._generation += 1
        gen = self._generation
        worker = SearchWorker(query, self.db_path, self.model_path, gen)
        worker.results_ready.connect(self._on_results)
        worker.start()
        self._worker = worker

    def _on_results(self, results: list, generation: int):
        if generation != self._generation:
            return

        self._results_list.clear()
        self._status.hide()

        if not results:
            self._sep.show()
            self._status.setText("No results")
            self._status.show()
            self._set_height(_SEP_H + _STATUS_H)
            return

        max_score = results[0]["score"]
        filtered = (
            [r for r in results if r["score"] >= SCORE_CUTOFF]
            if max_score >= SCORE_CUTOFF
            else results
        )

        total = len(filtered)
        for rank, r in enumerate(filtered):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, r["path"])
            item.setSizeHint(QSize(W, _ROW_H))
            self._results_list.addItem(item)
            self._results_list.setItemWidget(
                item,
                ResultItemWidget(r["filename"], r["path"], r["snippet"], rank, total),
            )

        list_h = min(total, _MAX_ROWS) * _ROW_H + _PAD_H
        self._results_list.setFixedHeight(list_h)
        self._sep.show()
        self._results_list.show()
        self._set_height(_SEP_H + list_h)

    def _open_result(self, item: QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            subprocess.Popen(["xdg-open", path])

    def show_and_focus(self):
        self._center_on_screen()
        self.show()
        self.raise_()
        self.activateWindow()
        self._input.setFocus()
        self._input.clear()
        self._results_list.clear()
        self._results_list.hide()
        self._sep.hide()
        self._status.hide()
        self._set_height(0)
