import importlib.util
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location('prepare_dataset', Path(__file__).parents[1] / 'scripts/prepare_dataset.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_data(tmp_path):
    model, original = tmp_path / 'small', tmp_path / 'original'
    for root in (model, original):
        for split in ('train', 'test'):
            scene = root / 'scenes_117' / split / 'room_0'
            (scene / 'traj_1').mkdir(parents=True)
            Image.new('RGB', (96, 96) if root == model else (256, 256)).save(scene / 'floorplan.png')
            np.save(scene / 'traj_1/traj_1.npy', np.array([[0., 0., 0.], [1., 1., 0.]]))
    (model / 'trav_maps').mkdir()
    np.save(model / 'trav_maps/floor_shapes.npy', {'room_0': [100, 100]})
    return model, original


@pytest.mark.parametrize('copy', [False, True])
def test_unified_layout_preserves_both_floorplans_and_references(tmp_path, copy):
    model, original = fixture_data(tmp_path)
    output = tmp_path / 'data'
    assert module.prepare(model, original, output, copy=copy) == 2
    scene = output / 'scenes_117/test/room_0'
    assert Image.open(scene / 'floorplan.png').size == (96, 96)
    assert Image.open(scene / 'planner_floorplan.png').size == (256, 256)
    np.testing.assert_array_equal(np.load(scene / 'traj_1/traj_1.npy'), np.load(original / 'scenes_117/test/room_0/traj_1/traj_1.npy'))
    assert (scene / 'traj_1').is_symlink() == (not copy)
    with pytest.raises(FileExistsError):
        module.prepare(model, original, output)


def test_conflicting_reference_rejected_before_output_created(tmp_path):
    model, original = fixture_data(tmp_path)
    np.save(original / 'scenes_117/test/room_0/traj_1/traj_1.npy', np.zeros((2, 3)))
    with pytest.raises(ValueError, match='Trajectory mismatch'):
        module.prepare(model, original, tmp_path / 'data')
    assert not (tmp_path / 'data').exists()
