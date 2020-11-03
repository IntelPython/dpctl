@REM call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"
@REM IF %ERRORLEVEL% NEQ 0 (
@REM 	echo "oneAPI compiler activation failed"
@REM 	exit /b 1
@REM )
@REM REM conda uses %ERRORLEVEL% but FPGA scripts can set it. So it should be reseted.
@REM set ERRORLEVEL=

@REM set "CC=clang-cl.exe"
@REM set "CXX=dpcpp.exe"

@REM rmdir /S /Q build_cmake
@REM mkdir build_cmake
@REM cd build_cmake

@REM set "DPCPP_ROOT=%ONEAPI_ROOT%\compiler\latest\windows"
@REM set "INSTALL_PREFIX=%cd%\..\install"

@REM rmdir /S /Q "%INSTALL_PREFIX%"

@REM cmake -G Ninja ^
@REM     -DCMAKE_BUILD_TYPE=Release ^
@REM     "-DCMAKE_INSTALL_PREFIX=%INSTALL_PREFIX%" ^
@REM     "-DCMAKE_PREFIX_PATH=%LIBRARY_PREFIX%" ^
@REM     "-DDPCPP_ROOT=%DPCPP_ROOT%" ^
@REM     "%SRC_DIR%\backends"
@REM IF %ERRORLEVEL% NEQ 0 exit /b 1

@REM ninja -n
@REM ninja install
@REM IF %ERRORLEVEL% NEQ 0 exit /b 1

@REM cd ..
@REM xcopy install\lib\*.lib dpctl /E /Y
@REM xcopy install\bin\*.dll dpctl /E /Y

@REM mkdir dpctl\include
@REM xcopy backends\include dpctl\include /E /Y


REM required by _sycl_core(dpctl)
set "DPPL_SYCL_INTERFACE_LIBDIR=dpctl"
set "DPPL_SYCL_INTERFACE_INCLDIR=dpctl\include"

"%PYTHON%" setup.py clean --all
"%PYTHON%" setup.py build install
IF %ERRORLEVEL% NEQ 0 exit /b 1
