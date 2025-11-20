#include "execution.hpp"

namespace backtest {

ExecutionEngine::ExecutionEngine() {}

void ExecutionEngine::submit_order(const Order& order) {
    active_orders_.push_back(order);
}

void ExecutionEngine::on_market_data(const Bar& bar) {
    process_orders(bar);
}

std::vector<Trade> ExecutionEngine::get_fills() {
    std::vector<Trade> fills = std::move(pending_fills_);
    pending_fills_.clear();
    return fills;
}

void ExecutionEngine::process_orders(const Bar& bar) {
    auto it = active_orders_.begin();
    while (it != active_orders_.end()) {
        if (it->instrument_id == bar.instrument_id) {
            bool filled = false;
            Price fill_price = 0.0;

            if (it->type == OrderType::Market) {
                // Fill at Open (simplified) or Close? Let's say Close for now as we process bar after it closes
                fill_price = bar.close;
                filled = true;
            } else if (it->type == OrderType::Limit) {
                if (it->side == Side::Buy && bar.low <= it->limit_price) {
                    fill_price = it->limit_price; // Optimistic fill
                    filled = true;
                } else if (it->side == Side::Sell && bar.high >= it->limit_price) {
                    fill_price = it->limit_price;
                    filled = true;
                }
            }

            if (filled) {
                Trade trade;
                trade.id = 0; // TODO: UUID
                trade.order_id = it->id;
                trade.instrument_id = it->instrument_id;
                trade.timestamp = bar.timestamp;
                trade.side = it->side;
                trade.quantity = it->quantity;
                trade.price = fill_price;
                trade.commission = 0.0; // TODO: Commission model

                pending_fills_.push_back(trade);
                it = active_orders_.erase(it);
                continue;
            }
        }
        ++it;
    }
}

} // namespace backtest
