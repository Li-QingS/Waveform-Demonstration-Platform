"""FDIDM 自适应纯函数测试（阶段 6c）。"""
import numpy as np

from waveform_sim.hardware.fdidm_adaptive import FDIDMAdaptiveMixin


def test_adaptive_qam_order():
    assert FDIDMAdaptiveMixin._adaptive_qam_order("QPSK") == 4
    assert FDIDMAdaptiveMixin._adaptive_qam_order("16QAM") == 16
    assert FDIDMAdaptiveMixin._adaptive_qam_order("64QAM") == 64
    assert FDIDMAdaptiveMixin._adaptive_qam_order("UNKNOWN") == 4


def test_adaptive_qfunc():
    q0 = float(FDIDMAdaptiveMixin._adaptive_qfunc(np.array([0.0]))[0])
    assert abs(q0 - 0.5) < 1e-9
    q = float(FDIDMAdaptiveMixin._adaptive_qfunc(np.array([1.96]))[0])
    assert 0.02 < q < 0.03


def test_finite_float_or_nan():
    assert FDIDMAdaptiveMixin._finite_float_or_nan(3.0) == 3.0
    assert np.isnan(FDIDMAdaptiveMixin._finite_float_or_nan(float("nan")))
    assert np.isnan(FDIDMAdaptiveMixin._finite_float_or_nan(float("inf")))


def test_adaptive_ser_from_symbol_nsr():
    obj = object.__new__(FDIDMAdaptiveMixin)
    ser = obj._adaptive_ser_from_symbol_nsr(np.array([100.0]), "QPSK")
    assert 0.0 <= float(ser) <= 1.0

