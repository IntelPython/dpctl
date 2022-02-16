from skbuild import setup

setup(
    name="sycl_gemm",
    version="0.0.1",
    description="an example of SYCL-powered Python package (with pybind11)",
    author="Intel Scripting",
    license="Apache 2.0",
    packages=["sycl_gemm"],
)
