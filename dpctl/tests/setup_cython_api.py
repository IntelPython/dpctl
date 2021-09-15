import setuptools

import dpctl

ext = setuptools.Extension(
    "_cython_api",
    ["_cython_api.pyx"],
    include_dirs=[dpctl.get_include()],
    language="c++",
)

setuptools.setup(name="_cython_api", version="0.0.0", ext_modules=[ext])
