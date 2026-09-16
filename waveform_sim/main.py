import sys
from pathlib import Path

# ``python waveform_sim/main.py`` is still supported for convenience.  Add the
# repository root only for that script form; package imports remain canonical.
if __package__ in (None, ""):
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from PyQt5.QtWidgets import QApplication
from waveform_sim.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 统一风格

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
