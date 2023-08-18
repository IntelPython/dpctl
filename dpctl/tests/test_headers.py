import pytest

import dpctl
import dpctl.tensor as dpt


@pytest.fixture(scope="session")
def dpctl_c_extension(tmp_path_factory):
    import os
    import os.path
    import shutil
    import subprocess
    import sys
    import sysconfig

    curr_dir = os.path.dirname(__file__)
    dr = tmp_path_factory.mktemp("_c_ext")
    for fn in ["_c_ext.c", "setup_c_ext.py"]:
        shutil.copy(
            src=os.path.join(curr_dir, fn),
            dst=dr,
            follow_symlinks=False,
        )
    res = subprocess.run(
        [sys.executable, "setup_c_ext.py", "build_ext", "--inplace"],
        cwd=dr,
        env=os.environ,
    )
    if res.returncode == 0:
        import glob
        from importlib.util import module_from_spec, spec_from_file_location

        sfx = sysconfig.get_config_vars()["EXT_SUFFIX"]
        pth = glob.glob(os.path.join(dr, "_c_ext*" + sfx))
        if not pth:
            pytest.fail("C extension was not built")
        spec = spec_from_file_location("_c_ext", pth[0])
        builder_module = module_from_spec(spec)
        spec.loader.exec_module(builder_module)
        return builder_module
    else:
        pytest.fail("C extension could not be built")


def test_c_headers(dpctl_c_extension):
    try:
        x = dpt.empty(10)
    except (dpctl.SyclDeviceCreationError, dpctl.SyclQueueCreationError):
        pytest.skip()

    assert dpctl_c_extension.is_usm_ndarray(x)
    assert dpctl_c_extension.usm_ndarray_ndim(x) == x.ndim
