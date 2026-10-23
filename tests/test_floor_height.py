import pytest
from eval.common import load_floor_height


def test_floor_heights_fail_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_floor_height('building', 1, tmp_path)
    folder = tmp_path / 'building'
    folder.mkdir()
    path = folder / 'floors.txt'
    path.write_text('0.008902\n3.118083\n')
    assert load_floor_height('building', 1, tmp_path) == 3.118083
    for floor in (-1, 2):
        with pytest.raises(ValueError):
            load_floor_height('building', floor, tmp_path)
    path.write_text('nan\n')
    with pytest.raises(ValueError):
        load_floor_height('building', 0, tmp_path)
