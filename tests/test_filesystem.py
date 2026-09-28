import pytest

from pcat.main import (
    _remove_empty_directories,
    make_directory,
    make_symlink,
    retr_pathcnfg,
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
