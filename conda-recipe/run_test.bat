call "%ONEAPI_ROOT%/compiler/latest/env/vars.bat"
IF ERRORLEVEL 1 exit 1

@echo on

"%PYTHON%" -c "import dppl"
IF ERRORLEVEL 1 exit 1

"%PYTHON%" -c "import dppl.ocldrv"
IF ERRORLEVEL 1 exit 1

"%PYTHON%" -m unittest -v dppl.tests
IF ERRORLEVEL 1 exit 1
