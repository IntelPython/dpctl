#!/bin/bash
set +xe

export CODE_COVERAGE=ON
python setup.py clean --all
python setup.py build develop
pytest --pyargs dpctl -vv

