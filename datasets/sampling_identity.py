"""Keep experiment sampling hashes stable when the raw data are relocated."""
from pathlib import Path


def sampling_root(root, identity_roots=None):
    if not identity_roots:
        return str(root)
    parts = Path(root).resolve().parts
    if any('without_obj' in p for p in parts):
        source = 'without_obj'
    elif any('with_obj' in p for p in parts):
        source = 'with_obj'
    else:
        raise ValueError('Sampling identity requires a with_obj/without_obj source directory: ' + str(root))
    return identity_roots[source]


def sampling_key(key, identity_roots=None):
    if not identity_roots:
        return key
    parts = key.split('::')
    parts[1] = sampling_root(parts[1], identity_roots)
    return '::'.join(parts)
