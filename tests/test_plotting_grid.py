import numpy as np

from pcat.plotting import plot_grid


def test_plot_grid_writes_one_and_three_parameter_pdfs(tmp_path):
    random = np.random.default_rng(123)
    one_path = plot_grid(
        tmp_path / "posterior",
        "one",
        random.normal(size=100),
        ("x",),
    )
    three_path = plot_grid(
        tmp_path / "posterior",
        "three",
        random.normal(size=(100, 3)),
        ("x", "y", "z"),
        truepara=(0.0, 0.0, 0.0),
        listvarbdraw=[(0.1, -0.1, 0.2)],
    )
    assert one_path.is_file()
    assert three_path.is_file()