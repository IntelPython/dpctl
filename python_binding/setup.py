from setuptools import setup
import versioneer

requirements = [
    'cffi>=1.0.0',
]

setup(
    name='dppy',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A lightweight Python wrapper for a subset of OpenCL and SYCL API.",
    license="BSD",
    author="Intel Corporation",
    url='https://github.intel.com/SAT/DP-Glue',
    packages=['dppy'],
    setup_requires=requirements,
    cffi_modules=[
        "./dppy/driverapi.py:ffi"
    ],
    install_requires=requirements,
    keywords='dppy',
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
