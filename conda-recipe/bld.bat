call "%ONEAPI_ROOT%compiler\latest\env\vars.bat"
IF ERRORLEVEL 1 exit 1
REM conda uses %ERRORLEVEL% but FPGA scripts can set it. So it should be reseted.
set ERRORLEVEL=

set "CC=dpcpp-cl.exe"
set "CXX=dpcpp-cl.exe"

rmdir /S /Q build_cmake
mkdir build_cmake
cd build_cmake

set "DPCPP_ROOT=%ONEAPI_ROOT%/compiler/latest/windows"

cmake -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    "-DCMAKE_INSTALL_PREFIX=%LIBRARY_PREFIX%" ^
    "-DCMAKE_PREFIX_PATH=%LIBRARY_PREFIX%" ^
    "-DDPCPP_ROOT=%DPCPP_ROOT%" ^
    "%SRC_DIR%/oneapi_wrapper"
IF ERRORLEVEL 1 exit 1

ninja -n
ninja install
IF ERRORLEVEL 1 exit 1

cd ..

REM required by dpglue
set "DP_GLUE_LIBDIR=%LIBRARY_PREFIX%/lib"
set "DP_GLUE_INCLDIR=%LIBRARY_PREFIX%/include"
set "OpenCL_LIBDIR=%DPCPP_ROOT%/lib"
REM required by oneapi_interface
set "DPPL_ONEAPI_INTERFACE_LIBDIR=%LIBRARY_PREFIX%/lib"
set "DPPL_ONEAPI_INTERFACE_INCLDIR=%LIBRARY_PREFIX%/include"

"%PYTHON%" setup.py clean --all
"%PYTHON%" setup.py build
"%PYTHON%" setup.py install
IF ERRORLEVEL 1 exit 1
