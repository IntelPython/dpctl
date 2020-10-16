import sys
from os.path import join, exists, abspath, dirname
from os import getcwd
from os import environ
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    import numpy as np
    import dpctl

    config = Configuration('', parent_package, top_path)

    oneapi_root = environ.get('ONEAPI_ROOT', None)
    if not oneapi_root:
        raise ValueError("ONEAPI_ROOT must be set, typical value is /opt/intel/oneapi")

    mkl_info = {
            'include_dirs': [join(oneapi_root, 'mkl', 'include')],
            'library_dirs': [join(oneapi_root, 'mkl', 'lib'), join(oneapi_root, 'mkl', 'lib', 'intel64')],
            'libraries': ['mkl_sycl', 'mkl_intel_ilp64', 'mkl_tbb_thread', 'mkl_core', 'tbb', 'iomp5']
    }

    mkl_include_dirs = mkl_info.get('include_dirs')
    mkl_library_dirs = mkl_info.get('library_dirs')
    mkl_libraries = mkl_info.get('libraries')

    pdir = dirname(__file__)
    wdir = join(pdir)

    eca = ['-Wall', '-Wextra', '-fsycl', '-fsycl-unnamed-lambda']

    config.add_extension(
        name='blackscholes_usm',
        sources=[
            join(pdir, 'blackscholes.pyx'),
            join(wdir, 'sycl_blackscholes.cpp'),
            join(wdir, 'sycl_blackscholes.hpp')
            ],
        include_dirs=[wdir, np.get_include(), dpctl.get_include()] + mkl_include_dirs,
        libraries=['sycl'] + mkl_libraries,
        runtime_library_dirs=mkl_library_dirs,
        extra_compile_args=eca, # + ['-O0', '-g', '-ggdb'],
        extra_link_args=['-fPIC'],
        language='c++'
    )

    config.ext_modules = cythonize(config.ext_modules, include_path=[pdir, wdir])
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
