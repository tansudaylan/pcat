import pytest

from pcat.main import make_directory, make_symlink, retr_pathcnfg


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