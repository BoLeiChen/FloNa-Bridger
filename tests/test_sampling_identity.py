import hashlib
from datasets.sampling_identity import sampling_key, sampling_root
from utils.category_validation import select_category_validation

ROOTS = {'without_obj': '/mnt/data/floverse_extracted/train_without_obj/HM3D',
         'with_obj': '/mnt/data/floverse_extracted/train_with_obj'}


def test_relocation_preserves_reference_hashes():
    for source, original in ROOTS.items():
        moved = '/new/machine/train_' + source + ('/HM3D' if source == 'without_obj' else '')
        before = 'traj::' + original + '::building_0::traj_3'
        after = 'traj::' + moved + '::building_0::traj_3'
        assert sampling_key(after, ROOTS) == before
        assert sampling_key(before, ROOTS) == before
        for suffix in ('', ':validation-tail'):
            assert hashlib.sha256((sampling_key(after, ROOTS) + suffix).encode()).digest() == hashlib.sha256((before + suffix).encode()).digest()
        assert sampling_root(moved, ROOTS) + '::building_0::traj_3::8' == original + '::building_0::traj_3::8'
        assert sampling_key(after) == after


def test_validation_selection_unchanged_after_relocation():
    def records(root):
        return [dict(key='traj::%s::building_0::traj_%d' % (root, i),
                     floor_key='floor::%s::building_0' % root, sample_count=10) for i in range(50)]
    original = records(ROOTS['without_obj'])
    moved = records('/new/machine/train_without_obj/HM3D')
    reference, audit = select_category_validation(original, {'building'}, 12, task='image')
    relocated, relocated_audit = select_category_validation(moved, {'building'}, 12, task='image', identity_roots=ROOTS)
    assert [sampling_key(x['key'], ROOTS) for x in relocated] == [x['key'] for x in reference]
    assert relocated_audit == audit


def test_sampling_identity_invalidates_old_validation_cache():
    from datasets.hypothesis_routes import route_signature
    legacy = route_signature({})
    assert route_signature(dict(sampling_identity_roots=ROOTS)) != legacy
    assert route_signature(dict(sampling_identity_roots=None)) == legacy
