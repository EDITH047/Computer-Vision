# Cloud Detection and Wind Speed Estimation

This project implements cloud detection and wind speed estimation from satellite images using C++ and OpenCV.

## Requirements
- CMake (version 3.10+)
- A C++17 compatible compiler (e.g., MSVC on Windows)
- OpenCV (version 4.x)

## OpenCV Installation on Windows

If you don't have OpenCV installed, you can download the pre-built binaries for Windows:
1. Go to the [OpenCV Releases Page](https://opencv.org/releases/)
2. Download the `Windows` package (e.g., `opencv-4.9.0-windows.exe`)
3. Extract it to `C:\opencv` (so that the build directory is at `C:\opencv\build`).

## Building the Project

### Using Command Line
Open a Developer Command Prompt (or PowerShell), navigate to this directory, and run:

```bat
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR="C:\opencv\build" ..
cmake --build . --config Release
```

*Note: Replace `C:\opencv\build` with the path where you installed OpenCV if it's different. Replace the generator string if you are using a different Visual Studio version or MinGW.*

### Running the Application

After building, you can run the application:

```bat
.\Release\CloudTracker.exe
```

A native file dialog will appear prompting you to select the first satellite image (T1) and the second image (T2). Then, you'll be asked to input the time difference in the console.

After pressing Enter, OpenCV windows will display the processing results:
1. Original Image 1
2. Binary Cloud Mask
3. Highlighted Clouds
4. Wind Direction Vectors
5. Estimated Wind Speed Map