"""硬件后端模块导入基线：四个 hardtest 模块可导入且主类存在。"""
import importlib

HARDTEST_CLASSES = {
    "waveform_sim.hardware.fdidm_hardtest": "FDIDMHardwareTest",
    "waveform_sim.hardware.ofdm_hardtest": "OfdmHardwareTx",
    "waveform_sim.hardware.otfs_hardtest": "OTFSHardwareTest",
    "waveform_sim.hardware.afdm_hardtest": "AFDMHardwareTest",
}


def test_hardtest_modules_import():
    for mod_name, cls_name in HARDTEST_CLASSES.items():
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, cls_name), f"{mod_name} 缺少 {cls_name}"

