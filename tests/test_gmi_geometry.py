import numpy as np
import pytest

from waveform_sim.gmi.geometry import (
    allocate_axis_counts,
    anisotropic_periodic_grid,
    fdit_generators,
    finite_difference_curvature,
    receiver_curvature,
)
from waveform_sim.gmi.model import ReceiverSpec
from scripts.reproduce_gmi_paper_example import reproduce_paper_example


SPECTRUM = [1.9, 1.4, 1.0, 0.6]


def test_curvature_loewner_bounds_and_full_receiver_zero():
    generators = fdit_generators(2, 2)
    diagonal = receiver_curvature(SPECTRUM, generators, 1.0, ReceiverSpec("diagonal"), 18)
    assert diagonal.valid
    assert diagonal.loewner_bounds_hold
    assert diagonal.curvature[0, 1] == pytest.approx(0.0, abs=1e-14)
    full = receiver_curvature(SPECTRUM, generators, 1.0, ReceiverSpec("full_matched", None), 18)
    assert full.valid
    assert np.allclose(full.curvature, 0.0)


@pytest.mark.parametrize(
    "rho, expected",
    [
        (1.0, (0.03221, 0.1045)),
        (10.0, (0.4821, 1.688)),
    ],
)
def test_paper_example_curvature(rho, expected):
    analysis = receiver_curvature(SPECTRUM, fdit_generators(2, 2), rho, ReceiverSpec("diagonal"), 18)
    assert analysis.curvature[0, 0] == pytest.approx(expected[0], rel=2e-3)
    assert analysis.curvature[1, 1] == pytest.approx(expected[1], rel=2e-3)
    assert analysis.curvature[1, 1] > analysis.curvature[0, 0]
    assert allocate_axis_counts(analysis.curvature, 36) == (4, 9)


def test_numerical_curvature_projects_small_negative_eigenvalues():
    result = finite_difference_curvature(lambda a, b: 1.0 - 2.0 * a * a - 8.0 * b * b + 1e-12 * a * b, step=1e-3)
    assert result.projected[0, 0] == pytest.approx(2.0, rel=1e-7)
    assert result.projected[1, 1] == pytest.approx(8.0, rel=1e-7)
    assert np.min(np.linalg.eigvalsh(result.projected)) >= -1e-14


def test_axis_allocation_and_periodic_grid_are_unique():
    assert allocate_axis_counts(np.diag([1.0, 16.0]), 16) == (2, 8)
    grid = anisotropic_periodic_grid(4, 4)
    assert len(grid) == 16
    assert len(set(grid)) == 16
    assert all(0.0 < alpha < 2.0 and 0.0 < beta < 2.0 for alpha, beta in grid)


def test_paper_example_sampled_regret_matches_reported_improvement():
    report = reproduce_paper_example()
    assert report["evaluation_grid"] == [33, 33]
    low_snr, high_snr = report["results"]
    for result in (low_snr, high_snr):
        assert (result["n_alpha"], result["n_beta"]) == (4, 9)
        assert result["quadratic_sampled_regret"] < result["equal_sampled_regret"]
    assert low_snr["relative_improvement_percent"] == pytest.approx(13.0348, abs=0.02)
    assert high_snr["relative_improvement_percent"] == pytest.approx(15.4439, abs=0.02)
