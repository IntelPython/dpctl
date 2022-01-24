
REM A workaround for activate-dpcpp.bat issue to be addressed in 2021.4
set "LIB=%BUILD_PREFIX%\Library\lib;%BUILD_PREFIX%\compiler\lib;%LIB%"
set "INCLUDE=%BUILD_PREFIX%\include;%INCLUDE%"

"%PYTHON%" setup.py clean --all
set "INSTALL_CMD=install -- -G Ninja -DDPCTL_DPCPP_HOME_DIR=%BUILD_PREFIX%\Library -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icx -DDPCTL_ENABLE_LO_PROGRAM_CREATION=ON"

if NOT "%WHEELS_OUTPUT_FOLDER%"=="" (
    rem Install and assemble wheel package from the build bits
    "%PYTHON%" setup.py %INSTALL_CMD% bdist_wheel
    if errorlevel 1 exit 1
    copy dist\dpctl*.whl %WHEELS_OUTPUT_FOLDER%
    if errorlevel 1 exit 1
) ELSE (
    rem Only install
    "%PYTHON%" setup.py %INSTALL_CMD%
    if errorlevel 1 exit 1
)
