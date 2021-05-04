#!/bin/bash
set +xe

# Check if level-zero headers are installed. Currently, works only for Ubuntu.
# Check https://dgpu-docs.intel.com/technologies/level-zero.html for details
# about what development package should be checked for different distros.
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$NAME" == "Ubuntu" ]]; then
        dpkg -s level-zero-dev
        if [[ $? == 0 ]]; then
            export HAS_LO_HEADERS=ON
        fi
    fi
fi
export CODE_COVERAGE=ON
python setup.py clean --all
python setup.py develop
pytest -q -ra --disable-warnings --cov dpctl --cov-report term-missing --pyargs dpctl -vv --cov-config=.coveragerc
