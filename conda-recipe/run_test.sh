#!/bin/bash

set -e

# Suppress error b/c it could fail on Ubuntu 18.04
source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh || true

${PYTHON} -c "import dpctl"
pytest -q -ra --disable-warnings --cov --cov-report term-missing --pyargs dpctl -vv
