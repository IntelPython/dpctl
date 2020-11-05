REM conda uses %ERRORLEVEL% but FPGA scripts can set it. So it should be reseted.
set ERRORLEVEL=

set "CC=clang-cl.exe"
set "CXX=dpcpp.exe"

rmdir /S /Q build_cmake
mkdir build_cmake
cd build_cmake

set "DPCPP_ROOT=%ONEAPI_ROOT%\compiler\latest\windows"
set "INSTALL_PREFIX=%cd%\..\install"

rmdir /S /Q "%INSTALL_PREFIX%"

cmake -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    "-DCMAKE_INSTALL_PREFIX=%INSTALL_PREFIX%" ^
    "-DCMAKE_PREFIX_PATH=%LIBRARY_PREFIX%" ^
    "-DDPCPP_ROOT=%DPCPP_ROOT%" ^
    "%SRC_DIR%\backends"
IF %ERRORLEVEL% NEQ 0 exit /b 1

ninja -n
ninja install
IF %ERRORLEVEL% NEQ 0 exit /b 1

cd ..
xcopy install\lib\*.lib dpctl /E /Y
xcopy install\bin\*.dll dpctl /E /Y

mkdir dpctl\include
xcopy backends\include dpctl\include /E /Y
