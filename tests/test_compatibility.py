import importlib
import io
import time
from types import SimpleNamespace

import numpy as np
import pytest

from pcat.main import (
    _resolve_population_elements,
    _should_plot_spatial_count_histograms,
    plot_genemaps,
    plot_scatcntp,
    retr_condcatl,
    retr_fromgdat,
    show_paragenrscalfull,
)


pcat_main = importlib.import_module("pcat.main")


def test_condensed_catalog_clusters_persistent_sources():
    element_names = ["xpos", "ypos", "flux"]
    posterior_catalogs = [
        [
            {
                "xpos": np.array([0.0, 1.0]),
                "ypos": np.array([0.0, 1.0]),
                "flux": np.array([2.0, 4.0]),
            }
        ],
        [
            {
                "xpos": np.array([0.01, 1.01]),
                "ypos": np.array([0.0, 1.0]),
                "flux": np.array([2.2, 3.8]),
            }
        ],
        [
            {
                "xpos": np.array([-0.01, 0.99]),
                "ypos": np.array([0.0, 1.0]),
                "flux": np.array([1.8, 4.2]),
            }
        ],
    ]
    gdat = SimpleNamespace(
        fitt=SimpleNamespace(
            indxpopl=np.array([0]),
            numbparagenrelem=len(element_names),
            namepara=SimpleNamespace(genrelem=[element_names]),
        ),
        indxsamptotl=np.arange(3),
        numbsamptotl=3,
        listpostindxelemfull=[[[0, 1]], [[0, 1]], [[0, 1]]],
        listpostdictelem=posterior_catalogs,
        typeverb=0,
        functime=time.time,
        anglassc=0.1,
        dictglob={},
        blimpara=SimpleNamespace(),
    )

    retr_condcatl(gdat)

    assert gdat.numbstkscond == 2
    assert gdat.numbprvlhigh == 2
    assert gdat.prvl == pytest.approx([1.0, 1.0])
    assert [
        entry["xpos"][0] for entry in gdat.dictglob["poststkscond"]
    ] == pytest.approx([0.0, 1.0])


def test_spatial_count_histograms_are_disabled_for_spectroscopy():
    assert not _should_plot_spatial_count_histograms(SimpleNamespace(anlytype="spec"))
    assert _should_plot_spatial_count_histograms(SimpleNamespace(anlytype="imag"))


def test_fixed_population_count_does_not_require_sampled_count_index():
    model = SimpleNamespace(indxpara=SimpleNamespace(numbelem=np.array([], dtype=int)))
    state = SimpleNamespace(numbelempopl=np.array([2]))

    count, indices = _resolve_population_elements(
        model, state, 0, np.arange(8, dtype=float), [[0, 1]]
    )

    assert count == 2
    np.testing.assert_array_equal(indices, [0, 1])


def test_retr_fromgdat_rejects_missing_uncertainty_map():
    gdat = SimpleNamespace(
        cntpdata=np.ones((1, 2, 1)),
        numbener=1,
        fitt=SimpleNamespace(),
    )

    with pytest.raises(AttributeError, match="cntpmodl is unavailable"):
        retr_fromgdat(
            gdat,
            None,
            "pdfn",
            "fitt",
            "cntpmodl",
            "post",
            strgmome="errr",
            indxvarb=[slice(None), slice(None), 0],
        )


def test_retr_fromgdat_rejects_missing_posterior_model():
    gdat = SimpleNamespace(
        cntpdata=np.array([4.0, 9.0]),
        numbener=1,
        fitt=SimpleNamespace(),
    )

    with pytest.raises(AttributeError, match="cntpmodl is unavailable"):
        retr_fromgdat(gdat, None, "pdfn", "fitt", "cntpmodl", "post")


def test_count_scatter_skips_missing_optional_posterior_model(capsys):
    gdat = SimpleNamespace(
        cntpdata=np.ones((1, 2, 1)),
        numbener=1,
        fitt=SimpleNamespace(),
    )

    plot_scatcntp(gdat, None, "pdfn", "fitt", "post", 0)

    assert "skipping count scatter plot" in capsys.readouterr().out


def test_count_plot_uses_spectral_axis_and_line_centers(monkeypatch):
    captured = {}
    monkeypatch.setitem(pcat_main.plt.rcParams, "text.usetex", False)
    model = SimpleNamespace(
        colr="tab:blue",
        colrelem=["tab:orange"],
        indxpopl=np.array([0]),
        typeelem=["lghtlinevoig"],
    )
    gdat = SimpleNamespace(
        cntpdata=np.ones((3, 1, 1)),
        numbener=3,
        numbpixl=1,
        typeexpr="fire",
        typedata="simu",
        bctrpara=SimpleNamespace(ener=np.array([4.0e5, 7.0e5, 1.0e6])),
        refr=SimpleNamespace(dictelem=[{"elin": np.array([[4.5e5, 8.0e5]])}]),
        lablener="E",
        strgenerunit=r"$\mu$m$^{-1}$",
        fitt=model,
        plotsize=4,
    )
    model_state = SimpleNamespace(
        cntpmodl=np.array([[[0.8]], [[1.2]], [[0.9]]]),
        dictelem=[{"elin": np.array([5.0e5, 9.0e5])}],
    )
    gdatmodi = SimpleNamespace(this=model_state)
    monkeypatch.setattr(
        pcat_main, "retr_plotpath", lambda *args, **kwargs: "unused.png"
    )
    monkeypatch.setattr(
        pcat_main,
        "savefigr",
        lambda gdat, gdatmodi, figure, path: captured.update(figure=figure),
    )

    plot_scatcntp(gdat, gdatmodi, "this", "fitt", "post", 0)

    axis = captured["figure"].axes[0]
    assert axis.get_xlabel() == r"$1 / \lambda$ [$\mu$m$^{-1}$]"
    assert axis.get_ylabel() == "Counts per spectral bin"
    assert [line.get_label() for line in axis.lines].count("True line") == 1
    assert [line.get_label() for line in axis.lines].count("Fitted line") == 1
    assert [
        line.get_xdata()[0] for line in axis.lines if line.get_linestyle() == "--"
    ] == pytest.approx([0.45, 0.8])
    assert [
        line.get_xdata()[0] for line in axis.lines if line.get_linestyle() == ":"
    ] == pytest.approx([0.5, 0.9])


def test_generic_map_skips_missing_optional_posterior_product(capsys):
    gdat = SimpleNamespace(
        cntpdata=np.ones((1, 2, 1)),
        numbener=1,
        fitt=SimpleNamespace(),
    )

    plot_genemaps(gdat, None, "pdfn", "fitt", "post", "cntpresi")

    assert "skipping cntpresi map" in capsys.readouterr().out


def test_show_paragenrscalfull_accepts_empty_element_indices(capsys):
    model_state = SimpleNamespace(
        paragenrunitfull=np.array([0.5]),
        paragenrscalfull=np.array([0.5]),
    )
    model = SimpleNamespace(
        indxpara=SimpleNamespace(genrfull=np.array([0])),
        indxpopl=np.array([0]),
        indxparagenrelemsing=[np.array([], dtype=int)],
        numbpopl=1,
        namepara=SimpleNamespace(genr=["value"]),
        scalpara=SimpleNamespace(genr=["self"]),
        this=model_state,
    )
    gdat = SimpleNamespace(fitt=model, booldiag=False)

    show_paragenrscalfull(gdat, None)

    assert "value" in capsys.readouterr().out


def test_proc_anim_uses_explicit_output_root(tmp_path, monkeypatch):
    captured = {}

    def fake_output_path(pathbase, run_name):
        captured["pathbase"] = pathbase
        captured["run_name"] = run_name
        return str(tmp_path) + "/"

    monkeypatch.setattr(pcat_main, "retr_pathoutpcnfg", fake_output_path)
    monkeypatch.setattr(
        pcat_main,
        "readfile",
        lambda path: SimpleNamespace(liststrgpdfn=[], liststrgfoldanim=[]),
    )
    monkeypatch.setattr(pcat_main, "open_narr", lambda path, mode: io.StringIO())

    pcat_main.proc_anim("demo", pathbase=tmp_path)

    assert captured == {"pathbase": tmp_path, "run_name": "demo"}


def test_proc_anim_preserves_numeric_sweep_order(tmp_path, monkeypatch):
    from PIL import Image

    output_path = tmp_path / "data" / "outp" / "demo"
    frame_path = tmp_path / "visuals" / "post" / "fram" / "maps"
    output_path.mkdir(parents=True)
    frame_path.mkdir(parents=True)
    Image.new("RGB", (2, 2), "blue").save(frame_path / "sample_swep000000010.png")
    Image.new("RGB", (2, 2), "red").save(frame_path / "sample_swep000000002.png")

    monkeypatch.setattr(
        pcat_main,
        "retr_pathoutpcnfg",
        lambda pathbase, run_name: str(output_path) + "/",
    )
    monkeypatch.setattr(
        pcat_main,
        "readfile",
        lambda path: SimpleNamespace(
            liststrgpdfn=["post"],
            pathvisu=str(tmp_path / "visuals") + "/",
            typefileplot="png",
        ),
    )

    pcat_main.proc_anim("demo", pathbase=tmp_path)

    animation_path = tmp_path / "visuals" / "post" / "anim" / "maps" / "sample.gif"
    with Image.open(animation_path) as animation:
        assert animation.n_frames == 2
        animation.seek(0)
        assert animation.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
        animation.seek(1)
        assert animation.convert("RGB").getpixel((0, 0)) == (0, 0, 255)


@pytest.mark.parametrize("typeexpr", ["chan", "ferm", "fire", "gmix", "HST_WFC3_IR"])
def test_sample_routes_image_experiments_through_image_initialization(
    typeexpr, monkeypatch
):
    captured = {}

    def fake_init_image(**configuration):
        captured["image"] = configuration
        return SimpleNamespace(typeexpr=typeexpr)

    monkeypatch.setattr(pcat_main, "init_image", fake_init_image)
    monkeypatch.setattr(pcat_main, "init", lambda configuration: configuration)

    result = pcat_main.sample(typeexpr=typeexpr)

    assert captured["image"]["typeexpr"] == typeexpr
    assert result == {"typeexpr": typeexpr}
