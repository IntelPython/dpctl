call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"
IF %ERRORLEVEL% NEQ 0 (
	echo "oneAPI compiler activation failed%"
	exit /b 1
)
REM conda uses %ERRORLEVEL% but FPGA scripts can set it. So it should be reseted.
set ERRORLEVEL=

@echo on

"%PYTHON%" -c "import dpctl"
IF %ERRORLEVEL% NEQ 0 exit /b 1

pytest --pyargs dpctl
IF %ERRORLEVEL% NEQ 0 exit /b 1
