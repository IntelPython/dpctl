#!/bin/bash

set -e

${PYTHON} -c "import dpctl; print(dpctl.__version__)"
${PYTHON} -m dpctl -f
# don't use coverage for Python 3.13 due to crashes related to
# Cython >= 3.1.0 and Python >= 3.13
# TODO: remove if crash is triaged
${PYTHON} -m pytest tests/my_test_file.py::test_specific_function -q -ra --disable-warnings --pyargs dpctl -vv 

