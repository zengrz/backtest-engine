# API Reference

## Python API

### `Backtester`
The main entry point.

- `add_strategy(strategy_cls)`: Adds a strategy class to the engine.
- `add_data(bars)`: Adds a list of `Bar` objects.
- `run()`: Starts the simulation.
- `get_portfolio()`: Returns the `Portfolio` object.

### `Strategy`
Base class for user strategies.

- `on_bar(bar)`: Callback for each bar.
- `buy(instrument_id, quantity, limit_price=0)`: Submit buy order.
- `sell(instrument_id, quantity, limit_price=0)`: Submit sell order.

### `Bar`
Market data structure.
- `timestamp`: int (ns)
- `instrument_id`: int
- `close`, `open`, `high`, `low`, `volume`: float

### `Portfolio`
- `get_cash()`: Returns available cash.
- `get_equity()`: Returns total equity.
- `get_position_quantity(instrument_id)`: Returns quantity for an instrument.
