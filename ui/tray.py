from pathlib import Path

from PyQt6.QtGui import QIcon, QPixmap, QColor, QPainter
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon


def _make_tray_icon() -> QIcon:
    """Create a simple colored circle icon if no asset is available."""
    assets_dir = Path(__file__).parent.parent / "assets"
    icon_path = assets_dir / "ferret.png"
    if icon_path.exists():
        return QIcon(str(icon_path))

    # Fallback: draw a small colored circle
    pixmap = QPixmap(22, 22)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#cba6f7"))
    painter.setPen(QColor("#1e1e2e"))
    painter.drawEllipse(2, 2, 18, 18)
    painter.end()
    return QIcon(pixmap)


class FerretTray(QSystemTrayIcon):
    def __init__(self, app: QApplication, on_settings=None, on_reindex=None,
                 on_force_reindex=None, on_quit=None):
        super().__init__(app)
        self.setIcon(_make_tray_icon())
        self.setToolTip("Ferret — Local Semantic Search")

        menu = QMenu()

        settings_action = menu.addAction("Settings")
        if on_settings:
            settings_action.triggered.connect(on_settings)

        menu.addSeparator()

        reindex_action = menu.addAction("Re-index")
        if on_reindex:
            reindex_action.triggered.connect(on_reindex)

        force_reindex_action = menu.addAction("Force Re-index (rebuild all)")
        if on_force_reindex:
            force_reindex_action.triggered.connect(on_force_reindex)

        menu.addSeparator()

        quit_action = menu.addAction("Quit")
        if on_quit:
            quit_action.triggered.connect(on_quit)
        else:
            quit_action.triggered.connect(app.quit)

        self.setContextMenu(menu)
        self.show()
