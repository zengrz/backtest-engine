#include "gpu_execution.hpp"
#include "gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

namespace backtest {

std::vector<double> GpuExecution::compute_sma(const std::vector<double>& prices, int period) {
    int n = prices.size();
    if (n == 0) return {};

    double* d_prices = nullptr;
    double* d_output = nullptr;
    size_t bytes = n * sizeof(double);

    cudaError_t err;

    err = cudaMalloc(&d_prices, bytes);
    if (err != cudaSuccess) {
        std::string error_msg = "cudaMalloc failed for d_prices: ";
        error_msg += cudaGetErrorString(err);
        throw std::runtime_error(error_msg);
    }

    err = cudaMalloc(&d_output, bytes);
    if (err != cudaSuccess) {
        cudaFree(d_prices);
        std::string error_msg = "cudaMalloc failed for d_output: ";
        error_msg += cudaGetErrorString(err);
        throw std::runtime_error(error_msg);
    }

    err = cudaMemcpy(d_prices, prices.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_prices);
        cudaFree(d_output);
        throw std::runtime_error("cudaMemcpy H2D failed");
    }

    cuda::launch_sma_kernel(d_prices, d_output, n, period);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_prices);
        cudaFree(d_output);
        std::string error_msg = "Kernel launch failed: ";
        error_msg += cudaGetErrorString(err);
        throw std::runtime_error(error_msg);
    }
    
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_prices);
        cudaFree(d_output);
        std::string error_msg = "Kernel execution failed: ";
        error_msg += cudaGetErrorString(err);
        throw std::runtime_error(error_msg);
    }

    std::vector<double> output(n);
    err = cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_prices);
    cudaFree(d_output);

    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy D2H failed");

    return output;
}

} // namespace backtest
