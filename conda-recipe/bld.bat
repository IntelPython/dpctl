call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"
if errorlevel 1 (
	echo "oneAPI compiler activation failed"
	exit /b 1
)

"%PYTHON%" setup.py clean --all
"%PYTHON%" setup.py install
if errorlevel 1 exit 1

rem Build wheel package
if NOT "%WHEELS_OUTPUT_FOLDER%"=="" (
    %PYTHON% setup.py bdist_wheel
    if errorlevel 1 exit 1
    copy dist\dpctl*.whl %WHEELS_OUTPUT_FOLDER%
    if errorlevel 1 exit 1
)
