call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"
IF %ERRORLEVEL% NEQ 0 (
	echo "oneAPI compiler activation failed"
	exit /b 1
)

"%PYTHON%" setup.py clean --all
"%PYTHON%" setup.py install
IF %ERRORLEVEL% NEQ 0 exit /b 1
