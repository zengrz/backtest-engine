#pragma once

namespace backtest {
namespace cuda {

void launch_sma_kernel(const double* d_prices, double* d_output, int n, int period);

} // namespace cuda
} // namespace backtest
