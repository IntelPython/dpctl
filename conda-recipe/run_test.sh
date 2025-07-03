#!/bin/bash

set -e

${PYTHON} -c "import dpctl; print(dpctl.__version__)"
${PYTHON} -m dpctl -f
if ${PYTHON} --version 2>&1 | grep -q '^Python 3\.13'; then
    ${PYTHON} -m pytest -q -ra --disable-warnings --pyargs dpctl -vv
else
    ${PYTHON} -m pytest -q -ra --disable-warnings --cov dpctl --cov-report term-missing --pyargs dpctl -vv
fi
