import multiprocessing
from pathlib import Path

import psutil
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Speed profiles
# ---------------------------------------------------------------------------

PROFILES = [
    {"label": "Safe",    "emoji": "Turtle",   "workers": 2, "description": "Low resource usage"},
    {"label": "Balanced","emoji": "Lightning", "workers": 4, "description": "Default, good for most laptops"},
    {"label": "Fast",    "emoji": "Rocket",    "workers": 8, "description": "For powerful desktops"},
    {"label": "Maximum", "emoji": "Fire",      "workers": None, "description": "Uses every available core"},
]

OCR_ENGINES = ["pytesseract", "PaddleOCR"]


def _ram_gb() -> float:
    return psutil.virtual_memory().total / (1024 ** 3)


def _ram_warning(workers: int) -> str:
    ram = _ram_gb()
    if workers >= 8 and ram < 8:
        return f"Warning: {workers} workers may be heavy on this system ({ram:.1f} GB RAM detected)"
    if workers >= 4 and ram < 4:
        return f"Warning: {workers} workers may be heavy on this system ({ram:.1f} GB RAM detected)"
    return f"System RAM: {ram:.1f} GB — looks good for {workers} workers"


class SettingsWindow(QDialog):
    def __init__(self, config: dict, on_save=None, parent=None):
        super().__init__(parent)
        self.config = dict(config)
        self.on_save = on_save
        self.setWindowTitle("Ferret Settings")
        self.setMinimumWidth(480)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # --- Indexed folders ---
        layout.addWidget(self._section_label("Indexed Folders"))

        self._folders_list = QListWidget()
        for folder in self.config.get("indexed_folders", []):
            self._folders_list.addItem(folder)
        layout.addWidget(self._folders_list)

        folder_btns = QHBoxLayout()
        add_btn = QPushButton("Add Folder")
        add_btn.clicked.connect(self._add_folder)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_folder)
        folder_btns.addWidget(add_btn)
        folder_btns.addWidget(remove_btn)
        layout.addLayout(folder_btns)

        # --- Indexing speed ---
        layout.addWidget(self._section_label("Indexing Speed Profile"))

        self._profile_combo = QComboBox()
        for p in PROFILES:
            workers = p["workers"] or multiprocessing.cpu_count()
            self._profile_combo.addItem(
                f"{p['label']} — {p['description']} ({workers} workers)"
            )
        # Set current selection from config
        current_workers = self.config.get("indexing_workers", 4)
        self._set_profile_by_workers(current_workers)
        self._profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        layout.addWidget(self._profile_combo)

        self._ram_label = QLabel()
        self._ram_label.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self._ram_label)
        self._on_profile_changed(self._profile_combo.currentIndex())

        # --- OCR engine ---
        layout.addWidget(self._section_label("OCR Engine"))

        self._ocr_combo = QComboBox()
        for engine in OCR_ENGINES:
            self._ocr_combo.addItem(engine)
        current_ocr = self.config.get("ocr_engine", "pytesseract")
        idx = OCR_ENGINES.index(current_ocr) if current_ocr in OCR_ENGINES else 0
        self._ocr_combo.setCurrentIndex(idx)
        layout.addWidget(self._ocr_combo)

        # --- Save / Cancel ---
        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
        return lbl

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Index")
        if folder:
            self._folders_list.addItem(folder)

    def _remove_folder(self):
        for item in self._folders_list.selectedItems():
            self._folders_list.takeItem(self._folders_list.row(item))

    def _set_profile_by_workers(self, workers: int):
        for i, p in enumerate(PROFILES):
            w = p["workers"] or multiprocessing.cpu_count()
            if w == workers or (p["workers"] is None and workers == multiprocessing.cpu_count()):
                self._profile_combo.setCurrentIndex(i)
                return
        # Default to Balanced
        self._profile_combo.setCurrentIndex(1)

    def _on_profile_changed(self, idx: int):
        profile = PROFILES[idx]
        workers = profile["workers"] or multiprocessing.cpu_count()
        self._ram_label.setText(_ram_warning(workers))

    def _save(self):
        folders = [
            self._folders_list.item(i).text()
            for i in range(self._folders_list.count())
        ]
        profile = PROFILES[self._profile_combo.currentIndex()]
        workers = profile["workers"] or multiprocessing.cpu_count()
        ocr_engine = self._ocr_combo.currentText()

        self.config["indexed_folders"] = folders
        self.config["indexing_workers"] = workers
        self.config["ocr_engine"] = ocr_engine

        if self.on_save:
            self.on_save(self.config)

        self.accept()
