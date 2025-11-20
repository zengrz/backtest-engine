import sys
import os

# Ensure the extension module can be found if running from source
# In a real install, this wouldn't be needed or would be handled by setup.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import _backtest_engine as _be
except ImportError as e:
    import sys
    import os
    cwd = os.getcwd()
    path = sys.path
    version = sys.version
    files = os.listdir(os.path.dirname(os.path.abspath(__file__)))
    raise ImportError(f"Failed to import _backtest_engine extension: {e}.\n"
                      f"Debug Info:\n"
                      f"  Python Version: {version}\n"
                      f"  CWD: {cwd}\n"
                      f"  Path: {path}\n"
                      f"  Files in {os.path.dirname(os.path.abspath(__file__))}: {files}\n"
                      f"Ensure the project is built and the extension is in the python path.") from e

# Export types for user convenience
Bar = _be.Bar
Side = _be.Side
Order = _be.Order
OrderType = _be.OrderType

class Strategy(_be.Strategy):
    """Base class for user strategies."""
    def __init__(self, engine=None):
        super().__init__()
        self.engine = engine

    def on_bar(self, bar: _be.Bar):
        """Called on every new bar."""
        pass

    def buy(self, instrument_id, quantity, limit_price=0.0):
        order = _be.Order()
        order.instrument_id = instrument_id
        order.quantity = quantity
        order.side = _be.Side.Buy
        order.type = _be.OrderType.Limit if limit_price > 0 else _be.OrderType.Market
        order.limit_price = limit_price
        self.engine.submit_order(order)

    def sell(self, instrument_id, quantity, limit_price=0.0):
        order = _be.Order()
        order.instrument_id = instrument_id
        order.quantity = quantity
        order.side = _be.Side.Sell
        order.type = _be.OrderType.Limit if limit_price > 0 else _be.OrderType.Market
        order.limit_price = limit_price
        self.engine.submit_order(order)

class Backtester:
    def __init__(self):
        self._loop = _be.EventLoop()
        self._strategies = []

    def add_strategy(self, strategy_cls):
        """Instantiates and adds a strategy."""
        strategy = strategy_cls(engine=self._loop)
        self._loop.add_strategy(strategy)
        self._strategies.append(strategy)
        return strategy

    def add_data(self, bars):
        """Adds historical data (list of Bar objects)."""
        self._loop.add_data(bars)

    def run(self):
        """Runs the simulation."""
        self._loop.run()

    def get_portfolio(self):
        return self._loop.get_portfolio()
    
    def compute_sma_gpu(self, prices, period):
        """Helper to use GPU SMA."""
        return _be.GpuExecution.compute_sma(prices, period)
