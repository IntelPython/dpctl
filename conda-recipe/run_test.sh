#!/bin/bash

set -e

# Suppress error b/c it could fail on Ubuntu 18.04
source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh || true

${PYTHON} -c "import dpctl"
${PYTHON} -m unittest -v dpctl.tests
