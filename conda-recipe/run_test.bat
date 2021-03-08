call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"
if errorlevel 1 (
	echo "oneAPI compiler activation failed%"
	exit /b 1
)
REM conda uses %ERRORLEVEL% but FPGA scripts can set it. So it should be reseted.
set ERRORLEVEL=

@echo on

"%PYTHON%" -c "import dpctl"
if errorlevel 1 exit 1

pytest -q -ra --disable-warnings --pyargs dpctl -vv
if errorlevel 1 exit 1
