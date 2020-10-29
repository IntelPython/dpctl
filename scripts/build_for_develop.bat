REM check if oneAPI has been activated, only try activating if not
dpcpp.exe --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    set ERRORLEVEL=
    call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"
    IF ERRORLEVEL 1 exit /b 1
)
REM conda uses %ERRORLEVEL% but FPGA scripts can set it. So it should be reseted.
set ERRORLEVEL=

rmdir /S /Q build_cmake
mkdir build_cmake

rmdir /S /Q install
mkdir install
cd install
set "INSTALL_PREFIX=%cd%"

cd ..\build_cmake

set "DPCPP_ROOT=%ONEAPI_ROOT%\compiler\latest\windows"
set NUMPY_INC=
for /f "delims=" %%a in ('%CONDA_PREFIX%\python.exe -c "import numpy; print(numpy.get_include())"') do @set NUMPY_INC=%%a 
set PYTHON_INC=
for /f "delims=" %%a in ('%CONDA_PREFIX%\python.exe -c "import distutils.sysconfig as sc; print(sc.get_python_inc())"') do @set PYTHON_INC=%%a 

if defined USE_GTEST (
    set "_GTEST_INCLUDE_DIR=%CONDA_PREFIX%\Library\include"
    set "_GTEST_LIB_DIR=%CONDA_PREFIX%\Library\lib"
) else (
    set "_GTEST_INCLUDE_DIR="
    set "_GTEST_LIB_DIR="
)
cmake -G Ninja ^
    -DCMAKE_BUILD_TYPE=Debug ^
    "-DCMAKE_CXX_FLAGS=-Wno-unused-function /EHa" ^
    "-DCMAKE_INSTALL_PREFIX=%INSTALL_PREFIX%" ^
    "-DCMAKE_PREFIX_PATH=%INSTALL_PREFIX%" ^
    "-DDPCPP_ROOT=%DPCPP_ROOT%" ^
    "-DCMAKE_C_COMPILER:PATH=%DPCPP_ROOT%\bin\clang-cl.exe" ^
    "-DCMAKE_CXX_COMPILER:PATH=%DPCPP_ROOT%\bin\dpcpp.exe" ^
    "-DGTEST_INCLUDE_DIR=%_GTEST_INCLUDE_DIR%" ^
    "-DGTEST_LIB_DIR=%_GTEST_LIB_DIR%" ^
    "-DPYTHON_INCLUDE_DIR=%PYTHON_INC%" ^
    "-DNUMPY_INCLUDE_DIR=%NUMPY_INC%" ^
    "%cd%\..\backends"
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
xcopy backends\include dpctl\include /E /Y


REM required by _sycl_core(dpctl)
set "DPPL_SYCL_INTERFACE_LIBDIR=dpctl"
set "DPPL_SYCL_INTERFACE_INCLDIR=dpctl\include"
set "CC=clang-cl.exe"
set "CXX=dpcpp.exe"

python setup.py clean --all
python setup.py build_ext --inplace develop
python -m unittest dpctl.tests
IF %ERRORLEVEL% NEQ 0 exit /b 1
