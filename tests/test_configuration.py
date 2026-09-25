from pathlib import Path


def test_run_name_does_not_control_sampler_behavior():
    source = (Path(__file__).parents[1] / 'pcat' / 'main.py').read_text()

    assert 'eval_lenscntpmodl' not in source