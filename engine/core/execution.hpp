#pragma once

#include "types.hpp"
#include <vector>
#include <queue>
#include <optional>

namespace backtest {

class ExecutionEngine {
public:
    ExecutionEngine();

    void submit_order(const Order& order);
    void on_market_data(const Bar& bar);
    
    // Returns filled trades since last call
    std::vector<Trade> get_fills();

private:
    std::vector<Order> active_orders_;
    std::vector<Trade> pending_fills_;
    
    void process_orders(const Bar& bar);
};

} // namespace backtest
