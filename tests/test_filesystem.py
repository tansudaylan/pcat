import pytest
import numpy as np
import tdpy

from pcat.main import (
    _remove_empty_directories,
    make_directory,
    make_symlink,
    readfile,
    retr_pathcnfg,
    retr_pathoutpcnfg,
    retr_pathplotcnfg,
    retr_pathrun,
    writfile,
)


def test_filesystem_helpers_support_spaces(tmp_path):
    directory = tmp_path / 'audit output'
    target = directory / 'target file.txt'
    link = directory / 'status link.txt'

    make_directory(directory)
    target.write_text('complete')
    make_symlink(target, link)

    assert link.read_text() == 'complete'


def test_retr_pathcnfg_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match='Invalid run tag'):
        retr_pathcnfg(tmp_path, '../outside')


def test_run_paths_are_self_contained_under_parent(tmp_path):
    run_root = tmp_path / 'pcat_runs' / 'gmix_demo'

    assert retr_pathrun(tmp_path, 'gmix_demo') == str(run_root)
    assert retr_pathoutpcnfg(tmp_path, 'gmix_demo') == (
        str(run_root / 'data' / 'outp' / 'gmix_demo') + '/'
    )
    assert retr_pathplotcnfg(tmp_path, 'gmix_demo') == (
        str(run_root / 'visuals') + '/'
    )


def test_run_paths_do_not_repeat_existing_runs_root(tmp_path):
    runs_root = tmp_path / 'pcat_runs'
    run_root = runs_root / 'gmix_demo'

    assert retr_pathrun(runs_root, 'gmix_demo') == str(run_root)


def test_run_paths_do_not_repeat_existing_run_root(tmp_path):
    run_root = tmp_path / 'pcat_runs' / 'gmix_demo'

    assert retr_pathrun(run_root, 'gmix_demo') == str(run_root)
    assert retr_pathoutpcnfg(run_root, 'gmix_demo') == (
        str(run_root / 'data' / 'outp' / 'gmix_demo') + '/'
    )
    assert retr_pathplotcnfg(run_root, 'gmix_demo') == (
        str(run_root / 'visuals') + '/'
    )


def test_remove_empty_directories_preserves_populated_branches(tmp_path):
    empty_leaf = tmp_path / 'optional' / 'unused'
    populated = tmp_path / 'results'
    empty_leaf.mkdir(parents=True)
    populated.mkdir()
    (populated / 'posterior.pdf').write_text('result')

    removed = _remove_empty_directories(tmp_path)

    assert str(empty_leaf) in removed
    assert not (tmp_path / 'optional').exists()
    assert (populated / 'posterior.pdf').is_file()


def test_state_round_trip_preserves_descriptive_string_arrays(tmp_path):
    state = tdpy.gdatstrt()
    state.nameproptype = np.array(['within_model', 'birth', 'death'])
    path = str(tmp_path / 'state')

    writfile(state, path)
    restored = readfile(path)

    np.testing.assert_array_equal(restored.nameproptype, state.nameproptype)
