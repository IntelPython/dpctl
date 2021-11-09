@REM check if oneAPI has been activated, only try activating if not
dpcpp.exe --version >nul 2>&1
if errorlevel 1 (
    set ERRORLEVEL=
    call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"
    if errorlevel 1 exit 1
)
@REM conda uses %ERRORLEVEL% but FPGA scripts can set it. So it should be reseted.
set ERRORLEVEL=

rmdir /S /Q build
mkdir build

rmdir /S /Q install
mkdir install
cd install
set "INSTALL_PREFIX=%cd%"

cd ..\build

set "DPCPP_HOME=%ONEAPI_ROOT%\compiler\latest\windows"

cmake -G Ninja ^
    -DCMAKE_BUILD_TYPE=Debug ^
    "-DCMAKE_CXX_FLAGS=-Wno-unused-function /EHsc" ^
    "-DCMAKE_INSTALL_PREFIX=%INSTALL_PREFIX%" ^
    "-DCMAKE_PREFIX_PATH=%INSTALL_PREFIX%" ^
    "-DDPCPP_INSTALL_DIR=%DPCPP_HOME%" ^
    "-DCMAKE_C_COMPILER:PATH=%DPCPP_HOME%\bin\icx.exe" ^
    "-DCMAKE_CXX_COMPILER:PATH=%DPCPP_HOME%\bin\dpcpp.exe" ^
    "-DCMAKE_LINKER:PATH=%DPCPP_HOME%\bin\lld-link.exe" ^
    "-DDPCTL_BUILD_CAPI_TESTS=ON" ^
    ".."
if errorlevel 1 exit 1

ninja -n
if errorlevel 1 exit 1
@REM ninja check
@REM IF %ERRORLEVEL% NEQ 0 exit /b 1
ninja install
if errorlevel 1 exit 1

cd ..
