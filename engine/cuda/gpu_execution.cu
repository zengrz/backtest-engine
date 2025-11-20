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
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed");

    err = cudaMalloc(&d_output, bytes);
    if (err != cudaSuccess) {
        cudaFree(d_prices);
        throw std::runtime_error("cudaMalloc failed");
    }

    err = cudaMemcpy(d_prices, prices.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_prices);
        cudaFree(d_output);
        throw std::runtime_error("cudaMemcpy H2D failed");
    }

    cuda::launch_sma_kernel(d_prices, d_output, n, period);

    std::vector<double> output(n);
    err = cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_prices);
    cudaFree(d_output);

    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy D2H failed");

    return output;
}

} // namespace backtest
