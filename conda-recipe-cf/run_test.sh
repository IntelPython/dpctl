#!/bin/bash

set -e

${PYTHON} -c "import dpctl; print(dpctl.__version__)"
${PYTHON} -m dpctl -f
${PYTHON} -m pytest -q -ra --disable-warnings --pyargs dpctl -vv
