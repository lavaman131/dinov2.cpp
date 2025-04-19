cmake_minimum_required(VERSION 3.12)
project(dinov2
        DESCRIPTION "Inference Vision Transformer (ViT) in plain C/C++ with ggml"
        HOMEPAGE_URL "https://github.com/staghado/vit.cpp")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set the compiler flags, modify here if needed!
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")


# Additional flags for C++ (the ffast-math is to be used with caution!)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")

# deps
add_subdirectory(ggml)

#set(OPENCV_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/opencv)
#
#include(ExternalProject)
#ExternalProject_Add(
#        opencv
#        GIT_REPOSITORY git@github.com:opencv/opencv.git
#        GIT_TAG 4.x
#        CMAKE_ARGS
#        -DBUILD_PERF_TESTS:BOOL=FALSE
#        -DBUILD_DOCS:BOOL=FALSE
#        -DBUILD_EXAMPLES:BOOL=FALSE
#        -DBUILD_TESTS:BOOL=FALSE
#        -DBUILD_SHARED_LIBS:BOOL=FALSE
#        -DBUILD_WITH_DEBUG_INFO=OFF
#        -DBUILD_PACKAGE:BOOL=OFF
#        -DCMAKE_BUILD_TYPE:STRING=Release
#        -DCMAKE_INSTALL_PREFIX:PATH=${OPENCV_INSTALL_DIR}
#)

#set(OpenCV_DIR ${OPENCV_INSTALL_DIR})
#find_package(OpenCV REQUIRED core videoio)
#
#`
# dinov2 executable
add_executable(${PROJECT_NAME} main.cpp dinov2.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} PUBLIC ggml)
target_include_directories(${PROJECT_NAME} ${OpenCV_INCLUDE_DIRS} PUBLIC .)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# quantize executable
option(BUILD_QUANTIZE "Build the quantization executable" ON)

if (BUILD_QUANTIZE)
    add_executable(quantize quantize.cpp)
    target_link_libraries(quantize PUBLIC ggml)
    target_include_directories(quantize PUBLIC .)
    target_compile_features(quantize PUBLIC cxx_std_17)
endif ()