#!/bin/bash

set -e

${PYTHON} -c "import dpctl; print(dpctl.__version__)"
${PYTHON} -c "import dpctl; dpctl.lsplatform(verbosity=2)"
${PYTHON} -m pytest -q -ra --disable-warnings -p no:faulthandler --cov dpctl --cov-report term-missing --pyargs dpctl -vv
