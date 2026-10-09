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

---

## Python Version

A complete Python port of this application is available in the [`python/`](python/) directory.
It is 100% functionally equivalent to the C++ version and requires no build step.

### Python Requirements
- Python 3.9 or newer
- `opencv-python >= 4.8.0`
- `numpy >= 1.24.0`
- `tkinter` (included in standard Python installations)

### Installing Python Dependencies

```bat
cd python
pip install -r requirements.txt
```

### Running the Python Application

```bat
cd python
python main.py
```

Or use the convenience launcher (installs dependencies automatically):

```bat
python\run.bat
```

A file dialog will appear to select T1 and T2 images, then the same three OpenCV windows are displayed:
1. **Original Image 1** — unmodified first frame
2. **Highlighted Clouds** — Jet colour-map heat-map keyed by wind speed + legend
3. **Wind Direction Vectors** — red arrows showing cloud motion direction

Statistics (cloud coverage %, average wind speed) are printed to the console.

### Running Automated Tests (headless)

```bat
cd python
python test_cloud_tracker.py
```

All 12 tests use synthetic images — no satellite images required.