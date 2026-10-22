import importlib.util
import io
from pathlib import Path
import tarfile
import pytest

spec = importlib.util.spec_from_file_location('extract', Path(__file__).parents[1] / 'scripts/extract_hf_dataset.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_archive(path, link, symbolic=False):
    with tarfile.open(path, 'w') as archive:
        item = tarfile.TarInfo('data/trav_maps/source.png')
        item.size = 4
        archive.addfile(item, io.BytesIO(b'map!'))
        item = tarfile.TarInfo('data/trav_maps/copy.png')
        item.type = tarfile.SYMTYPE if symbolic else tarfile.LNKTYPE
        item.linkname = link
        archive.addfile(item)


def test_extract_materializes_hard_links(tmp_path):
    archive = tmp_path / 'maps.tar'
    make_archive(archive, 'data/trav_maps/source.png')
    assert module.extract_archive(archive, tmp_path) == 2
    assert (tmp_path / 'data/trav_maps/copy.png').read_bytes() == b'map!'


@pytest.mark.parametrize('target,symbolic', [('../../outside', False), ('data/missing', False), ('data/trav_maps/source.png', True)])
def test_reject_unsafe_links(tmp_path, target, symbolic):
    archive = tmp_path / 'maps.tar'
    make_archive(archive, target, symbolic)
    with pytest.raises(ValueError):
        module.extract_archive(archive, tmp_path)
