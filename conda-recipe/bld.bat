
REM A workaround for activate-dpcpp.bat issue to be addressed in 2021.4
set "LIB=%CONDA_PREFIX%\Library\lib;%CONDA_PREFIX%\compiler\lib;%LIB%"

"%PYTHON%" setup.py clean --all
"%PYTHON%" setup.py install --sycl-compiler-prefix=%BUILD_PREFIX%\Library
if errorlevel 1 exit 1

rem Build wheel package
if NOT "%WHEELS_OUTPUT_FOLDER%"=="" (
    %PYTHON% setup.py bdist_wheel
    if errorlevel 1 exit 1
    copy dist\dpctl*.whl %WHEELS_OUTPUT_FOLDER%
    if errorlevel 1 exit 1
)
