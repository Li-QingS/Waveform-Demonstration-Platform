"""硬件后端兼容壳测试（阶段 6a）。"""
import importlib

import pytest


class _StubBackend:
    def get_status(self):
        return {"stub": True}


CASES = [
    ("waveform_sim.hardware.afdm_hardtest", "AFDMHardwareTest", "_LegacyAFDMHardwareTest"),
    ("waveform_sim.hardware.ofdm_hardtest", "OfdmHardwareTx", "_LegacyOfdmHardwareTx"),
]


@pytest.mark.parametrize("mod_name,shell_cls,legacy_cls", CASES)
def test_shell_delegates_to_backend(mod_name, shell_cls, legacy_cls):
    mod = importlib.import_module(mod_name)
    shell_cls = getattr(mod, shell_cls)
    legacy_cls = getattr(mod, legacy_cls)
    assert shell_cls.__name__ != legacy_cls.__name__
    shell = shell_cls(backend=_StubBackend())
    assert shell.get_status() == {"stub": True}
