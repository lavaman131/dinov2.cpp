cmake_minimum_required(VERSION 3.22.1)

set(OpenCV_INSTALL_DIR ${CMAKE_BINARY_DIR}/opencv-install)

include(ExternalProject)
ExternalProject_Add(OpenCV
        GIT_REPOSITORY "https://github.com/opencv/opencv.git"
        GIT_TAG "4.x"
        SOURCE_DIR ${CMAKE_BINARY_DIR}/opencv
        BINARY_DIR ${CMAKE_BINARY_DIR}/opencv-build
        CMAKE_ARGS
        -DBUILD_DOCS=FALSE
        -DBUILD_EXAMPLES=FALSE
        -DBUILD_TESTS=FALSE
        -DBUILD_opencv_apps=FALSE
        -DBUILD_SHARED_LIBS=TRUE
        -DWITH_CUDA=FALSE
        -DBUILD_JAVA=FALSE
        -DBUILD_opencv_python3=FALSE
        -DWITH_FFMPEG=FALSE
        -DBUILD_PERF_TESTS=FALSE
        -DOPENCV_ENABLE_NONFREE=TRUE
        -DBUILD_opencv_java=OFF
        -DCMAKE_INSTALL_PREFIX=${OpenCV_INSTALL_DIR}
)

set(OpenCV_DIR ${OpenCV_INSTALL_DIR}/lib/cmake/opencv4)
set(CMAKE_PREFIX_PATH ${OpenCV_INSTALL_DIR})
set(OpenCV_INCLUDE_DIRS ${OpenCV_INSTALL_DIR}/include/opencv4)
find_package(OpenCV REQUIRED NO_MODULE)