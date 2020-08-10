REM call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"
REM IF %ERRORLEVEL% NEQ 0 exit 1

"%PYTHON%" -c "import dppl"
IF %ERRORLEVEL% NEQ 0 exit 1

"%PYTHON%" -c "import dppl.ocldrv"
IF %ERRORLEVEL% NEQ 0 exit 1
