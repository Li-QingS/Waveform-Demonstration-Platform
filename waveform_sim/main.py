import os
import sys

# 保证 waveform_sim 包始终解析到本仓库（重构后含 core/ 子包），
# 避免被环境里其他旧版 waveform_sim 目录/已安装包遮蔽。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 统一风格

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
