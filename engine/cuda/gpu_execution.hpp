#pragma once

#include <vector>

namespace backtest {

class GpuExecution {
public:
    static std::vector<double> compute_sma(const std::vector<double>& prices, int period);
};

} // namespace backtest
