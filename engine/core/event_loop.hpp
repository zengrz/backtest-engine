#pragma once

#include "types.hpp"
#include "portfolio.hpp"
#include "execution.hpp"
#include <queue>
#include <functional>
#include <memory>

namespace backtest {

// Simple Strategy interface for callbacks
class Strategy {
public:
    virtual ~Strategy() = default;
    virtual void on_bar(const Bar& bar) = 0;
};

class EventLoop {
public:
    EventLoop();

    void add_strategy(std::shared_ptr<Strategy> strategy);
    void add_data(const std::vector<Bar>& bars);
    void run();

    Portfolio& get_portfolio();
    ExecutionEngine& get_execution();

    void submit_order(const Order& order);

private:
    Portfolio portfolio_;
    ExecutionEngine execution_;
    std::vector<std::shared_ptr<Strategy>> strategies_;
    
    // Priority queue for events (simplified to just Bars for now)
    // In a real system, this would be a polymorphic Event class
    struct Event {
        Timestamp timestamp;
        Bar bar;
        
        bool operator>(const Event& other) const {
            return timestamp > other.timestamp;
        }
    };

    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> event_queue_;
};

} // namespace backtest
