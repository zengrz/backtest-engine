cmake_minimum_required(VERSION 3.18)
project(backtest_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find dependencies
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/engine
    ${CUDAToolkit_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

# Collect sources
file(GLOB_RECURSE CORE_SOURCES "engine/core/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "engine/cuda/*.cu")
file(GLOB_RECURSE BINDING_SOURCES "engine/bindings/*.cpp")

# Create the core library (C++ + CUDA)
add_library(backtest_core SHARED
    ${CORE_SOURCES}
    ${CUDA_SOURCES}
)

target_link_libraries(backtest_core
    PRIVATE
    CUDA::cudart
)

# Create the Python extension module
pybind11_add_module(_backtest_engine
    ${BINDING_SOURCES}
)

target_link_libraries(_backtest_engine
    PRIVATE
    backtest_core
)

# Set output directory for the extension to the python folder
set_target_properties(_backtest_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
)
