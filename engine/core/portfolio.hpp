#pragma once

#include "types.hpp"
#include <unordered_map>
#include <vector>

namespace backtest {

class Portfolio {
public:
    Portfolio(double initial_cash);

    void on_fill(const Trade& trade);
    void update_market_price(InstrumentId instrument_id, Price price);

    double get_cash() const;
    double get_equity() const;
    double get_position_quantity(InstrumentId instrument_id) const;
    const std::unordered_map<InstrumentId, Position>& get_positions() const;

private:
    double cash_;
    double equity_;
    std::unordered_map<InstrumentId, Position> positions_;
    std::unordered_map<InstrumentId, Price> last_prices_;

    void recalculate_equity();
};

} // namespace backtest
