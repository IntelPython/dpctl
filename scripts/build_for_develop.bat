rmdir /S /Q build_cmake
mkdir build_cmake

rmdir /S /Q install
mkdir install
cd install
set "INSTALL_PREFIX=%cd%"

cd ..\build_cmake

set "DPCPP_ROOT=%BUILD_PREFIX%"

if defined USE_GTEST (
    set "_BUILD_CAPI_TEST=ON"
) else (
    set "_BUILD_CAPI_TEST=OFF"
)
cmake -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    "-DCMAKE_CXX_FLAGS=-Wno-unused-function /EHa" ^
    "-DCMAKE_INSTALL_PREFIX=%INSTALL_PREFIX%" ^
    "-DCMAKE_PREFIX_PATH=%INSTALL_PREFIX%" ^
    "-DDPCPP_ROOT=%DPCPP_ROOT%" ^
    "-DCMAKE_C_COMPILER:PATH=%DPCPP_ROOT%\bin\clang-cl.exe" ^
    "-DCMAKE_CXX_COMPILER:PATH=%DPCPP_ROOT%\bin\dpcpp.exe" ^
    "-DBUILD_CAPI_TESTS=%_BUILD_CAPI_TEST%" ^
    "%cd%\..\dpctl-capi"
IF %ERRORLEVEL% NEQ 0 exit /b 1

ninja -n
IF %ERRORLEVEL% NEQ 0 exit /b 1
if defined USE_GTEST (
    ninja check
    IF %ERRORLEVEL% NEQ 0 exit /b 1
)
ninja install
IF %ERRORLEVEL% NEQ 0 exit /b 1

cd ..
xcopy install\lib\*.lib dpctl /E /Y
xcopy install\bin\*.dll dpctl /E /Y

mkdir dpctl\include
xcopy dpctl-capi\include dpctl\include /E /Y


REM required by _sycl_core(dpctl)
set "DPCTL_SYCL_INTERFACE_LIBDIR=dpctl"
set "DPCTL_SYCL_INTERFACE_INCLDIR=dpctl\include"
set "CC=clang-cl.exe"
set "CXX=dpcpp.exe"

python setup.py clean --all
python setup.py build_ext --inplace develop
python -m unittest dpctl.tests
IF %ERRORLEVEL% NEQ 0 exit /b 1
