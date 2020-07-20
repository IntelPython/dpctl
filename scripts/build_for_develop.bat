SETLOCAL
set topdir=%CD%

IF EXIST build (
rmdir /Q /S build
)

if NOT ["%errorlevel%"]==["0"] (
    pause
    exit /b %errorlevel%
)

mkdir build
cd build
set WIN_SDK_LATEST=10.0.18362.0
set INSTALL_PREFIX=%topdir%\install
set DPCPP_ROOT=%ONEAPI_ROOT%\compiler\latest
REM set PATH=%PATH%;"C:\Program Files (x86)\Windows Kits\10\bin\10.0.18362.0\x64"

cmake                                                        ^
    -GNinja                                                  ^
    -DCMAKE_BUILD_TYPE=Release                               ^
    -DCMAKE_INSTALL_PREFIX=%INSTALL_PREFIX%                  ^
    -DCMAKE_PREFIX_PATH=%INSTALL_PREFIX%                     ^
    ..

if NOT ["%errorlevel%"]==["0"] (
    pause
    exit /b %errorlevel%
)

ninja -n
ninja install

if NOT ["%errorlevel%"]==["0"] (
    pause
    exit /b %errorlevel%
)

cd ..\python_binding

set DP_GLUE_LIBDIR=%INSTALL_PREFIX%\lib
set DP_GLUE_INCLDIR=%INSTALL_PREFIX%\include
set OpenCL_LIBDIR="C:\Program Files (x86)\IntelSWTools\sw_dev_tools\OpenCL\sdk\lib\x64"
set DPPL_ONEAPI_INTERFACE_LIBDIR=%INSTALL_PREFIX%\lib
set DPPL_ONEAPI_INTERFACE_INCLDIR=%INSTALL_PREFIX%\include

REM FIXME: How to pass this using setup.py? The fPIC flag is needed when
REM dpcpp compiles the Cython generated cpp file.
REM set CFLAGS=-fPIC
python setup.py build_ext --help-compiler
python setup.py clean --all
python setup.py build_ext --inplace
python setup.py develop

ENDLOCAL
