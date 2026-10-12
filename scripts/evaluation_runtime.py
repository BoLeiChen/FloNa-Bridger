"""Process-local EGL workaround; never installs/changes a system driver."""
import os
from pathlib import Path


def evaluation_environment(root):
    env=dict(os.environ,OMP_NUM_THREADS='2',MPLBACKEND='Agg',QT_QPA_PLATFORM='offscreen')
    vendor=Path(root)/'external/nvidia-595.84/runtime/egl_vendor.json'
    kernel=Path('/proc/driver/nvidia/version').read_text()
    if ' 595.84 ' in kernel and vendor.exists():
        env['__EGL_VENDOR_LIBRARY_FILENAMES']=str(vendor)
        env['LD_LIBRARY_PATH']=str(vendor.parent)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    else:
        env['__EGL_VENDOR_LIBRARY_FILENAMES']='/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
    env.pop('__EGL_VENDOR_LIBRARY_DIRS',None)
    return env
