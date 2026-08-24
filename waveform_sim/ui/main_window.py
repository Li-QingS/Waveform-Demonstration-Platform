from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QStatusBar, QLabel
from PyQt5.QtCore import Qt

from .fdidm_tab import FDIDMTab


def _optional_tab(import_path: str, class_name: str):
    try:
        module = __import__(import_path, fromlist=[class_name])
        return getattr(module, class_name)()
    except Exception as exc:
        label = QLabel(f"{class_name} 未加载：{exc}")
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        return label


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FDIDM软波形自适应演示平台 v2.3")
        self.setGeometry(80, 60, 1400, 900)
        self.setMinimumSize(1100, 700)
        self._init_ui()
        self._init_menu()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(4, 4, 4, 4)
        self.tabs = QTabWidget()
        self.tabs.setMovable(False)
        self.tabs.setTabsClosable(False)
        layout.addWidget(self.tabs)

        self.fdidm_tab = FDIDMTab()
        self.tabs.addTab(self.fdidm_tab, "软波形仿真")

        optional = [
            ("ui.ofdm_tab", "OfdmTab", "OFDM波形仿真"),
            ("ui.otfs_tab", "OTFSTab", "OTFS波形仿真"),
            ("ui.afdm_tab", "AfdmTab", "AFDM波形仿真"),
            ("ui.fdidm_hardware_test_tab", "FDIDMHardwareTestTab", "FDIDM硬件验证"),
            ("ui.hardware_test_tab", "HardwareTestTab", "硬件测评"),
            ("ui.waveform_compare_tab", "WaveformCompareTab", "波形对比分析"),
        ]
        for mod, cls, title in optional:
            try:
                self.tabs.addTab(_optional_tab(mod, cls), title)
            except Exception:
                pass

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("系统就绪：软波形仿真")
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _init_menu(self):
        menubar = self.menuBar()
        menubar.addMenu("文件(&F)")
        menubar.addMenu("视图(&V)")
        menubar.addMenu("帮助(&H)")

    def _on_tab_changed(self, index: int):
        self.status_bar.showMessage(f"当前页面：{self.tabs.tabText(index)}")
