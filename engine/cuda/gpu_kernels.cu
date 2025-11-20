#include <cuda_runtime.h>
#include "gpu_kernels.cuh"

__global__ void sma_kernel(const double* prices, double* output, int n, int period) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (idx < period - 1) {
        output[idx] = 0.0; // Not enough data
        return;
    }

    double sum = 0.0;
    for (int i = 0; i < period; ++i) {
        sum += prices[idx - period + 1 + i];
    }
    output[idx] = sum / period;
}

namespace backtest {
namespace cuda {

void launch_sma_kernel(const double* d_prices, double* d_output, int n, int period) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    sma_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_prices, d_output, n, period);
}

} // namespace cuda
} // namespace backtest
