
REM A workaround for activate-dpcpp.bat issue to be addressed in 2021.4
set "LIB=%BUILD_PREFIX%\Library\lib;%BUILD_PREFIX%\compiler\lib;%LIB%"
set "INCLUDE=%BUILD_PREFIX%\include;%INCLUDE%"

"%PYTHON%" setup.py clean --all
set "SKBUILD_ARGS=-- -G Ninja -DDPCTL_DPCPP_HOME_DIR=%BUILD_PREFIX%\Library -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icx -DDPCTL_ENABLE_L0_PROGRAM_CREATION=ON"
set "SYCL_INCLUDE_DIR_HINT=%BUILD_PREFIX%\Library\lib\clang\14.0.0"

set "PLATFORM_DIR=%PREFIX%\Library\share\cmake-3.22\Modules\Platform"
set "FN=Windows-IntelLLVM.cmake"

rem Save the original file, and copy patched file to
rem fix the issue with IntelLLVM integration with cmake on Windows
dir "%PLATFORM_DIR%\%FN%"
copy /Y "%PLATFORM_DIR%\%FN%" .
if errorlevel 1 exit 1
copy /Y .github\workflows\Windows-IntelLLVM.cmake "%PLATFORM_DIR%"
if errorlevel 1 exit 1

if NOT "%WHEELS_OUTPUT_FOLDER%"=="" (
    rem Install and assemble wheel package from the build bits
    "%PYTHON%" setup.py install bdist_wheel %SKBUILD_ARGS%
    if errorlevel 1 exit 1
    copy dist\dpctl*.whl %WHEELS_OUTPUT_FOLDER%
    if errorlevel 1 exit 1
) ELSE (
    rem Only install
    "%PYTHON%" setup.py install %SKBUILD_ARGS%
    if errorlevel 1 exit 1
)

rem copy back
copy /Y "%FN%" "%PLATFORM_DIR%"
if errorlevel 1 exit 1
