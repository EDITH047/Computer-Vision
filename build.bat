@echo off
echo Creating build directory...
if not exist build mkdir build
cd build

echo.
echo Please ensure OpenCV is installed in C:\opencv\build
echo If it is installed elsewhere, you will need to edit this script.
echo.

echo Configuring CMake project...
cmake -DOpenCV_DIR="C:\opencv\build" -DOpenCV_RUNTIME=vc16 -DOpenCV_ARCH=x64 ..

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed. 
    echo Please make sure you have CMake installed and OpenCV extracted to C:\opencv.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Building the project...
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo Build failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Build successful! The executable is located in build\Release\CloudTracker.exe (or build\CloudTracker.exe depending on your compiler).
pause
