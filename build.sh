#!/bin/bash

# Build script for backtest-engine
# This script configures and builds the project with CUDA 12.9 and Python 3.11

set -e  # Exit on error

echo "=========================================="
echo "Building backtest-engine"
echo "=========================================="

# Configuration
CUDA_PATH="/usr/local/cuda-12.9"
PYTHON_EXEC="/home/rey/github/backtest-engine/.conda/bin/python"
PYBIND11_DIR="/home/rey/github/backtest-engine/.conda/lib/python3.11/site-packages/pybind11/share/cmake/pybind11"

# Clean and create build directory
echo "Cleaning build directory..."
rm -rf build
mkdir build
cd build

# Configure with CMake
echo "Configuring with CMake..."
PATH="${CUDA_PATH}/bin:$PATH" cmake \
    -DCMAKE_CUDA_COMPILER="${CUDA_PATH}/bin/nvcc" \
    -DPython3_EXECUTABLE="${PYTHON_EXEC}" \
    -Dpybind11_DIR="${PYBIND11_DIR}" \
    ..

# Build
echo "Building..."
make -j$(nproc)

echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
echo "To verify the build, run:"
echo "  ${PYTHON_EXEC} python/test_build.py"
echo ""
echo "To run benchmarks:"
echo "  ${PYTHON_EXEC} examples/benchmark.py"
echo "  ${PYTHON_EXEC} examples/benchmark_cpu.py"
