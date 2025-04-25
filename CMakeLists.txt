cmake_minimum_required(VERSION 3.22.1)
project(dinov2
        DESCRIPTION "Inference DINOv2 in plain C/C++ with ggml"
        HOMEPAGE_URL "https://github.com/lavaman131/dinov2.cpp")

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set the compiler flags, modify here if needed!
# -Ofast is to be used with caution! if doesn't work, try -O3
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")

# deps
add_subdirectory(ggml)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI

# OPTIONAL: set(OpenCV_DIR {PATH TO BUILD FOLDER HERE})
find_package(OpenCV REQUIRED)

if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please set OpenCV_DIR to the directory containing OpenCVConfig.cmake.")
endif ()


# dinov2 executable
add_executable(inference inference.cpp dinov2.cpp)
target_link_libraries(inference PRIVATE ${OpenCV_LIBS} PUBLIC ggml)
target_include_directories(inference PUBLIC .)

option(BUILD_QUANTIZE "Build the quantization executable" ON)
option(BUILD_REALTIME "Build the realtime executable" ON)


if (BUILD_REALTIME)
    add_executable(realtime realtime.cpp dinov2.cpp)
    target_link_libraries(realtime PRIVATE ${OpenCV_LIBS} PUBLIC ggml)
    target_include_directories(realtime PUBLIC .)
endif ()

if (BUILD_QUANTIZE)
    add_executable(quantize quantize.cpp dinov2.cpp)
    target_link_libraries(quantize PRIVATE ${OpenCV_LIBS} PUBLIC ggml)
    target_include_directories(quantize PUBLIC .)
endif ()