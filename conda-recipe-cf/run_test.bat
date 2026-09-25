@echo on

"%PYTHON%" -c "import dpctl; print(dpctl.__version__)"
if %ERRORLEVEL% neq 0 exit 1

"%PYTHON%" -m dpctl -f
if %ERRORLEVEL% neq 0 exit 1

python -m pytest -q -ra --disable-warnings --pyargs dpctl -vv
if %ERRORLEVEL% neq 0 exit 1
