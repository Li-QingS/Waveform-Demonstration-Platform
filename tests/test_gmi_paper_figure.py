import json
from pathlib import Path

import matplotlib.image as mpimg
import pytest

from scripts import reproduce_gmi_paper_example as paper


@pytest.fixture
def paper_report():
    return {
        "charged_samples": 18,
        "evaluation_grid": [33, 33],
        "library_budget": 36,
        "radius": 0.04,
        "results": [
            {
                "curvature_diagonal": [0.03220622596986383, 0.10453764852870832],
                "equal_grid": [6, 6],
                "equal_sampled_regret": 6.076725167919461e-06,
                "n_alpha": 4,
                "n_beta": 9,
                "quadratic_sampled_regret": 5.284633923741122e-06,
                "relative_improvement_percent": 13.034837388400993,
                "rho": 1.0,
            },
            {
                "curvature_diagonal": [0.48212242293407304, 1.688059778250783],
                "equal_grid": [6, 6],
                "equal_sampled_regret": 9.641603397825271e-05,
                "n_alpha": 4,
                "n_beta": 9,
                "quadratic_sampled_regret": 8.152564397978512e-05,
                "relative_improvement_percent": 15.443893908586013,
                "rho": 10.0,
            },
        ],
        "spectrum": [1.9, 1.4, 1.0, 0.6],
        "version": 1,
    }


def test_render_paper_figure_creates_nonempty_hd_png(tmp_path, paper_report):
    target = tmp_path / "nested" / "paper.png"

    returned = paper.render_paper_figure(paper_report, target)

    assert returned == target.resolve()
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert target.stat().st_size > 20_000
    image = mpimg.imread(target)
    assert image.shape[:2] == (1200, 1920)
    assert float(image[..., :3].std()) > 0.05


def test_render_rejects_incomplete_report_without_partial_file(tmp_path):
    target = tmp_path / "paper.png"

    with pytest.raises(ValueError, match="exactly two"):
        paper.render_paper_figure({"results": []}, target)

    assert not target.exists()


def test_cli_generates_default_figure_and_preserves_json(monkeypatch, tmp_path, paper_report, capsys):
    target = tmp_path / "default.png"
    json_target = tmp_path / "result.json"
    monkeypatch.setattr(paper, "DEFAULT_FIGURE", target)
    monkeypatch.setattr(paper, "reproduce_paper_example", lambda _points: paper_report)

    assert paper.main(["--json", str(json_target)]) == 0

    output = capsys.readouterr().out
    assert f"Figure: {target.resolve()}" in output
    assert target.exists()
    assert json.loads(json_target.read_text(encoding="utf-8")) == paper_report


def test_cli_no_figure_does_not_invoke_renderer(monkeypatch, tmp_path, paper_report, capsys):
    monkeypatch.setattr(paper, "DEFAULT_FIGURE", tmp_path / "must-not-exist.png")
    monkeypatch.setattr(paper, "reproduce_paper_example", lambda _points: paper_report)
    monkeypatch.setattr(
        paper,
        "render_paper_figure",
        lambda *_args: pytest.fail("renderer must not run with --no-figure"),
    )

    assert paper.main(["--no-figure"]) == 0

    output = capsys.readouterr().out
    assert json.loads(output) == paper_report
    assert not paper.DEFAULT_FIGURE.exists()


def test_cli_rejects_figure_and_no_figure_together():
    with pytest.raises(SystemExit) as error:
        paper.main(["--figure", "x.png", "--no-figure"])

    assert error.value.code == 2
