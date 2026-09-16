from pathlib import Path
from core.language import is_rtl

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QSize, QUrl
from PyQt6.QtGui import QKeySequence, QShortcut, QDesktopServices
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

SCORE_CUTOFF = 0.25
FONT = "'Helvetica Neue', 'Ubuntu', 'Noto Sans', sans-serif"
W = 680          # fixed window width

_INPUT_H  = 66   # height of just the input row
_SEP_H    = 1    # hairline separator
_ROW_H    = 84   # room for the matching passage and its source
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
                 rank: int, total: int, parent=None, result=None):
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
        name.setTextFormat(Qt.TextFormat.PlainText)
        name.setLayoutDirection(Qt.LayoutDirection.RightToLeft if is_rtl(filename) else Qt.LayoutDirection.LeftToRight)
        name.setStyleSheet(
            f"color: rgba(255,255,255,{ta});"
            f"font-size: {'14px' if rank == 0 else '13px'};"
            f"font-weight: {'500' if rank == 0 else '400'};"
            f"font-family: {FONT};"
        )

        sub_text = (snippet[:95].replace("\n", " ").strip()
                    if snippet else str(Path(path).parent))
        sub = QLabel(sub_text)
        sub.setTextFormat(Qt.TextFormat.PlainText)
        sub.setAlignment(Qt.AlignmentFlag.AlignRight if is_rtl(snippet) else Qt.AlignmentFlag.AlignLeft)
        sub.setLayoutDirection(Qt.LayoutDirection.RightToLeft if is_rtl(snippet) else Qt.LayoutDirection.LeftToRight)
        sub.setStyleSheet(
            f"color: rgba(255,255,255,{sa});"
            f"font-size: 12px; font-weight: 400;"
            f"font-family: {FONT};"
        )

        col.addWidget(name)
        col.addWidget(sub)
        result = result or {}
        routes = {"filename": "Name", "fts": "Keyword", "semantic": "Meaning", "document": "Document topic", "normalized": "Arabic / normalized",
                  "filename_normalized": "Name / normalized",
                  "reranker": "Reranked",
                  "doctype": "Document type"}
        details = [routes.get(route, route) for route in result.get("matched_by", [])]
        if result.get("page") and ext == ".pdf":
            details.append(f"Page {result['page']}")
        elif result.get("start_line") and ext in {".txt", ".md"}:
            details.append(f"Line {result['start_line']}")
        details.append(str(Path(path).parent))
        if len(result.get("copies", [])) > 1:
            details.append(f"{len(result['copies'])} copies")
        source = QLabel(" · ".join(details))
        source.setTextFormat(Qt.TextFormat.PlainText)
        source.setStyleSheet(f"color: #92929d; font-size: 10px; font-family: {FONT};")
        col.addWidget(source)
        self.setToolTip("\n".join(result.get("copies", [path])) + "\n\n" + snippet)
        layout.addLayout(col, 1)


class SearchWorker(QThread):
    results_ready = pyqtSignal(list, int)
    warning_ready = pyqtSignal(str, int)

    def __init__(self, query: str, db_path: str, model_path: str, generation: int):
        super().__init__()
        self.query = query
        self.db_path = db_path
        self.model_path = model_path
        self.generation = generation
        self.reranker_path = None
        self.rerank_minimum = None
        self.calibration_path = None

    def run(self):
        try:
            from core.searcher import search
            diagnostics = []
            # Paint inexpensive exact/keyword results while models are loading.
            preliminary = search(self.query, self.db_path, top_k=8, model_path=self.model_path, mode="keyword")
            if preliminary and not self.isInterruptionRequested():
                self.results_ready.emit(preliminary, self.generation)
            if self.isInterruptionRequested():
                return
            results = search(self.query, self.db_path, top_k=8, model_path=self.model_path,
                             reranker_path=self.reranker_path, rerank_minimum=self.rerank_minimum,
                             calibration_path=self.calibration_path,
                             on_candidates=lambda values: self.results_ready.emit(values, self.generation),
                             cancelled=self.isInterruptionRequested,
                             diagnostics=diagnostics)
        except Exception as e:
            print(f"[searchbar] Search error: {e}")
            diagnostics = [str(e)]
            results = []
        self.results_ready.emit(results, self.generation)
        if diagnostics:
            self.warning_ready.emit(" · ".join(diagnostics), self.generation)


class SearchBar(QWidget):
    def __init__(self, db_path: str, model_path: str = "~/ferret/models/bge-small-en"):
        super().__init__()
        self.db_path = db_path
        self.model_path = model_path
        self._worker: SearchWorker | None = None
        self._generation = 0
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(200)
        self._debounce.timeout.connect(self._run_search)
        self._pending_query = ""
        self._closing = False
        self.indexing_status = None
        self.reranker_path = None
        self.rerank_minimum = None
        self._setup_ui()
        self._setup_hotkeys()
        QApplication.instance().aboutToQuit.connect(self._shutdown)

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
        self._input.setPlaceholderText("Search by meaning…  type:pdf  in:~/Documents")
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
        QShortcut(QKeySequence(Qt.Key.Key_Down), self, lambda: self._select_result(1))
        QShortcut(QKeySequence(Qt.Key.Key_Up), self, lambda: self._select_result(-1))
        self._input.returnPressed.connect(self._open_selected)
        QShortcut(QKeySequence("Ctrl+Shift+C"), self, self._copy_selected_path)

    def _select_result(self, delta):
        count = self._results_list.count()
        if count and self._results_list.isVisible():
            self._results_list.setCurrentRow(
                (self._results_list.currentRow() + delta) % count
            )

    def _open_selected(self):
        item = self._results_list.currentItem()
        if item and self._results_list.isVisible():
            self._open_result(item)

    def _copy_selected_path(self):
        item = self._results_list.currentItem()
        if item and self._results_list.isVisible():
            QApplication.clipboard().setText(item.data(Qt.ItemDataRole.UserRole))

    def _center_on_screen(self):
        screen = QApplication.primaryScreen().geometry()
        self.move(
            screen.center().x() - W // 2,
            screen.center().y() - _INPUT_H // 2,   # center on just the input
        )

    def _on_text_changed(self, text: str):
        text = text.strip()
        # Invalidate immediately, including while the debounce is waiting and
        # when the input is cleared. A running query must never repaint it.
        self._generation += 1
        if self._worker is not None:
            self._worker.requestInterruption()
        self._pending_query = text
        self._input.setLayoutDirection(Qt.LayoutDirection.RightToLeft if is_rtl(text) else Qt.LayoutDirection.LeftToRight)
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
        if not query or self._closing:
            return
        if self._worker is not None:
            return  # Keep only the newest pending query; one inference at a time.
        gen = self._generation
        worker = SearchWorker(query, self.db_path, self.model_path, gen)
        worker.reranker_path = self.reranker_path
        worker.rerank_minimum = self.rerank_minimum
        worker.calibration_path = getattr(self, "calibration_path", None)
        worker.results_ready.connect(self._on_results)
        worker.warning_ready.connect(self._on_warning)
        worker.finished.connect(self._worker_finished)
        self._worker = worker
        worker.start()

    def _worker_finished(self):
        worker = self._worker
        self._worker = None
        if worker is not None:
            generation = worker.generation
            worker.deleteLater()
            if generation != self._generation and not self._debounce.isActive():
                self._run_search()

    def _shutdown(self):
        self._closing = True
        self._debounce.stop()
        if self._worker is not None:
            self._worker.requestInterruption()
            self._worker.wait()

    def _on_warning(self, message, generation):
        if generation != self._generation or self._closing:
            return
        self._status.setText("Search is incomplete — " + message[:95])
        self._status.setToolTip(message)
        self._status.show()
        self._set_height(_SEP_H + _STATUS_H + (
            self._results_list.height() if not self._results_list.isHidden() else 0))

    def _on_results(self, results: list, generation: int):
        if generation != self._generation or self._closing:
            return

        self._results_list.clear()
        self._status.hide()

        if not results:
            self._sep.show()
            status = self.indexing_status() if self.indexing_status else None
            if status and not status.idle:
                self._status.setText(f"Indexing in progress · {status.pending} files queued")
            else:
                self._status.setText("No results")
            self._status.show()
            self._set_height(_SEP_H + _STATUS_H)
            return

        # RRF scores are relative ranks, not a probability of relevance.
        filtered = results

        total = len(filtered)
        for rank, r in enumerate(filtered):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, r["path"])
            item.setSizeHint(QSize(W, _ROW_H))
            self._results_list.addItem(item)
            self._results_list.setItemWidget(
                item,
                ResultItemWidget(r["filename"], r["path"], r["snippet"], rank, total, result=r),
            )

        list_h = min(total, _MAX_ROWS) * _ROW_H + _PAD_H
        self._results_list.setFixedHeight(list_h)
        self._sep.show()
        self._results_list.show()
        self._results_list.setCurrentRow(0)
        self._set_height(_SEP_H + list_h)

    def _open_result(self, item: QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            if QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
                self.hide()

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
