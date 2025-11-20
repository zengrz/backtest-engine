#pragma once

#include <cstdint>
#include <string>
#include <chrono>

namespace backtest {

using Timestamp = std::int64_t; // Nanoseconds since epoch
using Price = double;
using Quantity = double;
using InstrumentId = std::int32_t;

enum class Side {
    Buy,
    Sell
};

enum class OrderType {
    Market,
    Limit
};

struct Bar {
    Timestamp timestamp;
    InstrumentId instrument_id;
    Price open;
    Price high;
    Price low;
    Price close;
    Quantity volume;
};

struct Order {
    std::int64_t id;
    InstrumentId instrument_id;
    Timestamp timestamp;
    Side side;
    OrderType type;
    Quantity quantity;
    Price limit_price; // 0 for Market
};

struct Trade {
    std::int64_t id;
    std::int64_t order_id;
    InstrumentId instrument_id;
    Timestamp timestamp;
    Side side;
    Quantity quantity;
    Price price;
    Price commission;
};

struct Position {
    InstrumentId instrument_id;
    Quantity quantity;
    Price average_cost;
};

} // namespace backtest
