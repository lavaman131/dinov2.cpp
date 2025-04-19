cmake_minimum_required(VERSION 3.22.1)

if (NOT OpenCV_TAG)
    set(OpenCV_TAG "4.x")
endif ()

set(OpenCV_INSTALL_DIR ${CMAKE_BINARY_DIR}/opencv-install)
set(CMAKE_PREFIX_PATH ${OpenCV_INSTALL_DIR})
find_package(OpenCV)

if (${OpenCV_FOUND})
    message(STATUS "OpenCV found: ${OpenCV_DIR}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else ()
    message(STATUS "OpenCV not found. Downloading and building from
  source...")
    include(ExternalProject)
    ExternalProject_Add(OpenCV
            GIT_REPOSITORY "https://github.com/opencv/opencv.git"
            GIT_TAG "${OpenCV_TAG}"
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
    set(CMAKE_PREFIX_PATH ${OpenCV_INSTALL_DIR})
    set(OpenCV_DIR ${OpenCV_INSTALL_DIR})
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_CONFIG_PATH})
endif ()