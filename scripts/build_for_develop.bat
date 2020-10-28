call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"
IF ERRORLEVEL 1 exit /b 1
REM conda uses %ERRORLEVEL% but FPGA scripts can set it. So it should be reseted.
set ERRORLEVEL=

set "CC=clang-cl.exe"
set "CXX=dpcpp.exe"

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

cmake -G Ninja ^
    -DCMAKE_BUILD_TYPE=Debug ^
    "-DCMAKE_CXX_FLAGS=-Wno-unused-function" ^
    "-DCMAKE_INSTALL_PREFIX=%INSTALL_PREFIX%" ^
    "-DCMAKE_PREFIX_PATH=%INSTALL_PREFIX%" ^
    "-DDPCPP_ROOT=%DPCPP_ROOT%" ^
    "-DPYTHON_INCLUDE_DIR=%PYTHON_INC%" ^
    "-DGTEST_INCLUDE_DIR=%CONDA_PREFIX%\Library\include" ^
    "-DGTEST_LIB_DIR=%CONDA_PREFIX%\Library\lib" ^
    "-DNUMPY_INCLUDE_DIR=%NUMPY_DIR%" ^
    "%cd%\..\backends"
IF %ERRORLEVEL% NEQ 0 exit /b 1

ninja -n 
IF %ERRORLEVEL% NEQ 0 exit /b 1
ninja check
IF %ERRORLEVEL% NEQ 0 exit /b 1
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

python setup.py clean --all
python setup.py build_ext --inplace develop
python -m unittest dpctl.tests
IF %ERRORLEVEL% NEQ 0 exit /b 1
