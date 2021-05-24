#!/bin/bash
set +xe

python setup.py clean --all
python setup.py develop --coverage=True
pytest -q -ra --disable-warnings --cov dpctl --cov-report term-missing --pyargs dpctl -vv
