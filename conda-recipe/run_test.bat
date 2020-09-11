call "%ONEAPI_ROOT%/compiler/latest/env/vars.bat"
IF ERRORLEVEL 1 exit 1
REM conda uses %ERRORLEVEL% but FPGA scripts can set it. So it should be reseted.
set ERRORLEVEL=

@echo on

"%PYTHON%" -c "import dppl"
IF ERRORLEVEL 1 exit 1

"%PYTHON%" -c "import dppl.ocldrv"
IF ERRORLEVEL 1 exit 1

"%PYTHON%" -m unittest -v dppl.tests
IF ERRORLEVEL 1 exit 1
