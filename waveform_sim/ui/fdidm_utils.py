"""FDIDM 页数据类与静态助手（从 fdidm_tab.py 搬移，阶段 8）。"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class _CurveSpec:
    name: str
    alpha: float
    beta: float
    pen: object


def alpha_ser_floor(num_frames: int, m_subcarriers: int, n_symbols: int) -> float:
    """Lower plotting bound for the left-bottom α-SER curves.

    The left-bottom α-SER plot is a Monte-Carlo-oriented comparison.
    Even when the selected value mode is theory, values below the
    Monte-Carlo resolvability of the selected α-curve frame count are
    clipped to the rule-of-three zero-error upper bound 3/(frames*M*N).
    This prevents the plot from displaying artificial 1e-12 values that
    cannot be supported by a 1000-frame run.
    """
    frames = int(max(1, num_frames))
    m = int(max(1, m_subcarriers))
    n = int(max(1, n_symbols))
    return float(3.0 / (frames * m * n))


def merged_curve_specs(raw_specs):
    """Merge curves that are mathematically the same α/β point.

    This prevents the right-bottom plot from drawing two permanently
    overlapping curves, e.g. when the manual current point is exactly OFDM
    (0,0), or when the searched optimum is exactly OTFS (1,1).
    """
    merged = OrderedDict()
    for label, a, b in raw_specs:
        key = (round(float(a), 10), round(float(b), 10))
        if key not in merged:
            merged[key] = {"labels": [str(label)], "alpha": float(a), "beta": float(b)}
        else:
            merged[key]["labels"].append(str(label))
    out = []
    for item in merged.values():
        labels = item["labels"]
        name = " / ".join(labels)
        out.append((name, item["alpha"], item["beta"]))
    return out


def copy_kwargs_with(base_kwargs, **updates):
    data = dict(base_kwargs)
    data.update(updates)
    return data
