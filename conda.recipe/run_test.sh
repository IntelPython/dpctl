#!/bin/bash

set -e

source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh

${PYTHON} -c "import dppy"
${PYTHON} -c "import dppy.ocldrv"
