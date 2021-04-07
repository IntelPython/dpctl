#!/bin/bash

set -e

# Suppress error b/c it could fail on Ubuntu 18.04
source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh || true

${PYTHON} -c "import dpctl"
coverage run -m --source=dpctl --branch --omit=*/dpctl/tests/*,*/dpctl/_version.py pytest -q -ra --disable-warnings --pyargs dpctl -vv && coverage report -m
