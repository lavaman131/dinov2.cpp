cmake_minimum_required(VERSION 3.22.1)
project(dinov2
        DESCRIPTION "Inference Vision Transformer (ViT) in plain C/C++ with ggml"
        HOMEPAGE_URL "https://github.com/staghado/vit.cpp")

set(CMAKE_CXX_STANDARD 17)

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

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
include(External-OpenCV)


# Additional flags for C++ (the ffast-math is to be used with caution!)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")

# deps
add_subdirectory(ggml)

# dinov2 executable
add_executable(${PROJECT_NAME} main.cpp dinov2.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} PUBLIC ggml)
target_include_directories(${PROJECT_NAME} PUBLIC . ${OpenCV_INCLUDE_DIRS})

add_dependencies(${PROJECT_NAME} OpenCV)

# quantize executable
option(BUILD_QUANTIZE "Build the quantization executable" ON)

if (BUILD_QUANTIZE)
    add_executable(quantize quantize.cpp)
    target_link_libraries(quantize PUBLIC ggml)
    target_include_directories(quantize PUBLIC .)
endif ()