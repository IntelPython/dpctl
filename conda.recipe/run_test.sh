#!/bin/bash

set -e

source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh

${PYTHON} -c "import dppl"
${PYTHON} -c "import dppl.ocldrv"
