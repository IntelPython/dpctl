from setuptools import setup
import versioneer
from Cython.Build import cythonize
import numpy as np

requirements = [
    'cffi>=1.0.0',
]

dppy_oneapi_interface_lib     = os.environ['DPPY_ONEAPI_INTERFACE_LIBDIR']
dppy_oneapi_interface_include = os.environ['DPPY_ONEAPI_INTERFACE_INCLDIR']

def getpyexts():
    libs = [-lrt, -lDDPYOneapiInterface]
    exts = cythonize(Extension('_oneapi_interface',
                               [os.path.abspath('dppy_rt/oneapi_interface.pyx'),],
                                depends=[dppy_oneapi_interface_include,],
                                include_dirs=[np.get_include()],
                                libraries=libs,
                                library_dirs=[dppy_oneapi_interface_lib],
                                language='c++'))

setup(
    name='dppy_rt',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A lightweight Python wrapper for a subset of OpenCL and SYCL API.",
    license="Apache 2.0",
    author="Intel Corporation",
    url='https://github.intel.com/SAT/dppy',
    packages=['dppy_rt'],
    ext_modules = getpyexts(),
    setup_requires=requirements,
    cffi_modules=[
        "./dppy/driverapi.py:ffi"
    ],
    install_requires=requirements,
    keywords='dppy_rt',
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
