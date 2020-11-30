call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"
IF %ERRORLEVEL% NEQ 0 (
	echo "oneAPI compiler activation failed"
	exit /b 1
)

"%PYTHON%" setup.py clean --all
"%PYTHON%" setup.py install
IF %ERRORLEVEL% NEQ 0 exit /b 1

rem Build wheel package
if NOT "%WHEELS_OUTPUT_FOLDER%"=="" (
    %PYTHON% setup.py bdist_wheel
    if errorlevel 1 exit 1
    copy dist\dpctl*.whl %WHEELS_OUTPUT_FOLDER%
    if errorlevel 1 exit 1
)
