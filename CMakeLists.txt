cmake_minimum_required(VERSION 3.10)
project(CloudTracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Attempt to find OpenCV. 
# If OpenCV is installed in C:/opencv, set OpenCV_DIR to C:/opencv/build
# You can pass -DOpenCV_DIR="C:/opencv/build" during cmake configuration.
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "src/*.h")

add_executable(CloudTracker ${SOURCES} ${HEADERS})

target_link_libraries(CloudTracker ${OpenCV_LIBS})

# We are using Windows native file dialog, so we might need Comdlg32
if(WIN32)
    target_link_libraries(CloudTracker Comdlg32)
endif()
