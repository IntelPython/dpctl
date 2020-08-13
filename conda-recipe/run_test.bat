call "%ONEAPI_ROOT%/compiler/latest/env/vars.bat"
IF %ERRORLEVEL% NEQ 0 exit 1

@echo on

"%PYTHON%" -c "import dppl"
IF %ERRORLEVEL% NEQ 0 exit 1

"%PYTHON%" -c "import dppl.ocldrv"
IF %ERRORLEVEL% NEQ 0 exit 1

"%PYTHON%" -m unittest -v dppl.tests
IF %ERRORLEVEL% NEQ 0 exit 1
