#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "core/types.hpp"
#include "core/portfolio.hpp"
#include "core/execution.hpp"
#include "core/event_loop.hpp"
#include "cuda/gpu_execution.hpp"

namespace py = pybind11;
using namespace backtest;

// Trampoline for Strategy
class PyStrategy : public Strategy {
public:
    using Strategy::Strategy; // Inherit constructors

    void on_bar(const Bar& bar) override {
        PYBIND11_OVERRIDE_PURE(
            void,      // Return type
            Strategy,  // Parent class
            on_bar,    // Name of function in C++ (must match Python name)
            bar        // Argument(s)
        );
    }
};

PYBIND11_MODULE(_backtest_engine, m) {
    m.doc() = "High-performance Backtest Engine";

    // Enums
    py::enum_<Side>(m, "Side")
        .value("Buy", Side::Buy)
        .value("Sell", Side::Sell)
        .export_values();

    py::enum_<OrderType>(m, "OrderType")
        .value("Market", OrderType::Market)
        .value("Limit", OrderType::Limit)
        .export_values();

    // Structs
    py::class_<Bar>(m, "Bar")
        .def(py::init<>())
        .def_readwrite("timestamp", &Bar::timestamp)
        .def_readwrite("instrument_id", &Bar::instrument_id)
        .def_readwrite("open", &Bar::open)
        .def_readwrite("high", &Bar::high)
        .def_readwrite("low", &Bar::low)
        .def_readwrite("close", &Bar::close)
        .def_readwrite("volume", &Bar::volume);

    py::class_<Order>(m, "Order")
        .def(py::init<>())
        .def_readwrite("id", &Order::id)
        .def_readwrite("instrument_id", &Order::instrument_id)
        .def_readwrite("timestamp", &Order::timestamp)
        .def_readwrite("side", &Order::side)
        .def_readwrite("type", &Order::type)
        .def_readwrite("quantity", &Order::quantity)
        .def_readwrite("limit_price", &Order::limit_price);

    py::class_<Trade>(m, "Trade")
        .def_readwrite("id", &Trade::id)
        .def_readwrite("order_id", &Trade::order_id)
        .def_readwrite("instrument_id", &Trade::instrument_id)
        .def_readwrite("timestamp", &Trade::timestamp)
        .def_readwrite("side", &Trade::side)
        .def_readwrite("quantity", &Trade::quantity)
        .def_readwrite("price", &Trade::price)
        .def_readwrite("commission", &Trade::commission);

    py::class_<Position>(m, "Position")
        .def_readwrite("instrument_id", &Position::instrument_id)
        .def_readwrite("quantity", &Position::quantity)
        .def_readwrite("average_cost", &Position::average_cost);

    // Classes
    py::class_<Portfolio>(m, "Portfolio")
        .def("get_cash", &Portfolio::get_cash)
        .def("get_equity", &Portfolio::get_equity)
        .def("get_position_quantity", &Portfolio::get_position_quantity)
        .def("get_positions", &Portfolio::get_positions);

    py::class_<ExecutionEngine>(m, "ExecutionEngine");

    py::class_<Strategy, PyStrategy, std::shared_ptr<Strategy>>(m, "Strategy")
        .def(py::init<>())
        .def("on_bar", &Strategy::on_bar);

    py::class_<EventLoop>(m, "EventLoop")
        .def(py::init<>())
        .def("add_strategy", &EventLoop::add_strategy)
        .def("add_data", &EventLoop::add_data)
        .def("run", &EventLoop::run)
        .def("get_portfolio", &EventLoop::get_portfolio, py::return_value_policy::reference)
        .def("get_execution", &EventLoop::get_execution, py::return_value_policy::reference)
        .def("submit_order", &EventLoop::submit_order);

    // GPU Utils
    py::class_<GpuExecution>(m, "GpuExecution")
        .def_static("compute_sma", &GpuExecution::compute_sma);
}
