#include "portfolio.hpp"

namespace backtest {

Portfolio::Portfolio(double initial_cash)
    : cash_(initial_cash), equity_(initial_cash) {}

void Portfolio::on_fill(const Trade& trade) {
    cash_ -= trade.price * trade.quantity;
    cash_ -= trade.commission;

    auto& pos = positions_[trade.instrument_id];
    pos.instrument_id = trade.instrument_id;
    
    // Update average cost (simplified)
    if (pos.quantity == 0) {
        pos.average_cost = trade.price;
    } else if ((pos.quantity > 0 && trade.quantity > 0) || (pos.quantity < 0 && trade.quantity < 0)) {
        double total_cost = pos.quantity * pos.average_cost + trade.quantity * trade.price;
        pos.average_cost = total_cost / (pos.quantity + trade.quantity);
    }
    // If reducing position, average cost doesn't change (FIFO/LIFO matters here but keeping simple)
    
    pos.quantity += trade.quantity;

    if (pos.quantity == 0) {
        positions_.erase(trade.instrument_id);
    }

    recalculate_equity();
}

void Portfolio::update_market_price(InstrumentId instrument_id, Price price) {
    last_prices_[instrument_id] = price;
    recalculate_equity();
}

double Portfolio::get_cash() const {
    return cash_;
}

double Portfolio::get_equity() const {
    return equity_;
}

double Portfolio::get_position_quantity(InstrumentId instrument_id) const {
    auto it = positions_.find(instrument_id);
    if (it != positions_.end()) {
        return it->second.quantity;
    }
    return 0.0;
}

const std::unordered_map<InstrumentId, Position>& Portfolio::get_positions() const {
    return positions_;
}

void Portfolio::recalculate_equity() {
    equity_ = cash_;
    for (const auto& [id, pos] : positions_) {
        if (last_prices_.count(id)) {
            equity_ += pos.quantity * last_prices_.at(id);
        } else {
            // Fallback to average cost if no market price available? 
            // Or just don't update equity component for this asset.
            // For now, let's use average cost as a proxy if price is missing, or 0.
            equity_ += pos.quantity * pos.average_cost; 
        }
    }
}

} // namespace backtest
