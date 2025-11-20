#include "event_loop.hpp"
#include <iostream>

namespace backtest {

EventLoop::EventLoop() : portfolio_(100000.0) {} // Default 100k cash

void EventLoop::add_strategy(std::shared_ptr<Strategy> strategy) {
    strategies_.push_back(strategy);
}

void EventLoop::add_data(const std::vector<Bar>& bars) {
    for (const auto& bar : bars) {
        event_queue_.push({bar.timestamp, bar});
    }
}

void EventLoop::run() {
    while (!event_queue_.empty()) {
        Event event = event_queue_.top();
        event_queue_.pop();

        // 1. Update Market Data in Portfolio (for equity calc)
        portfolio_.update_market_price(event.bar.instrument_id, event.bar.close);

        // 2. Process Orders / Execution
        execution_.on_market_data(event.bar);
        auto fills = execution_.get_fills();
        for (const auto& fill : fills) {
            portfolio_.on_fill(fill);
        }

        // 3. Notify Strategies
        for (auto& strategy : strategies_) {
            strategy->on_bar(event.bar);
        }
    }
}

Portfolio& EventLoop::get_portfolio() {
    return portfolio_;
}

ExecutionEngine& EventLoop::get_execution() {
    return execution_;
}

void EventLoop::submit_order(const Order& order) {
    // Basic validation could go here
    execution_.submit_order(order);
}

} // namespace backtest
