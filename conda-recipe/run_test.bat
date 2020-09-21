call "%ONEAPI_ROOT%/compiler/latest/env/vars.bat"
IF ERRORLEVEL 1 exit 1
REM conda uses %ERRORLEVEL% but FPGA scripts can set it. So it should be reseted.
set ERRORLEVEL=

@echo on

"%PYTHON%" -c "import dpctl"
IF %ERRORLEVEL% NEQ 0 exit 1

"%PYTHON%" -c "import dpctl.ocldrv"
IF %ERRORLEVEL% NEQ 0 exit 1

"%PYTHON%" -m unittest -v dpctl.tests
IF %ERRORLEVEL% NEQ 0 exit 1
