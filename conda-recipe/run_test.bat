@echo on

"%PYTHON%" -c "import dpctl; print(dpctl.__version__)"
if errorlevel 1 exit 1

"%PYTHON%" -m dpctl -f
if errorlevel 1 exit 1

python -m pytest -q -ra --disable-warnings --pyargs dpctl -vv
if errorlevel 1 exit 1
