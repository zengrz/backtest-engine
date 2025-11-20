import sys
import os
import time
import random

# Add python directory to path
sys.path.append('../python')

try:
    from backtester import Backtester, Strategy, Bar
    print("Successfully imported Backtester")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def calculate_sma_python(prices, period):
    """Pure Python SMA calculation"""
    sma_values = []
    for i in range(len(prices)):
        if i < period - 1:
            sma_values.append(0.0)
            continue
        
        window = prices[i - period + 1 : i + 1]
        sma = sum(window) / period
        sma_values.append(sma)
    return sma_values

class SimpleStrategy(Strategy):
    """Simple strategy for benchmarking"""
    def __init__(self, engine):
        super().__init__(engine)
        self.bar_count = 0
        
    def on_bar(self, bar):
        self.bar_count += 1

def main():
    print("=" * 60)
    print("BACKTEST ENGINE PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Test 1: SMA Calculation
    print("\n[Test 1] SMA Calculation Performance")
    print("-" * 60)
    DATA_SIZE = 100000
    PERIOD = 50
    print(f"Dataset: {DATA_SIZE:,} prices, Period: {PERIOD}")
    prices = [100.0 + random.random() * 100 for _ in range(DATA_SIZE)]
    
    print("\nPure Python implementation...")
    start = time.time()
    sma_python = calculate_sma_python(prices, PERIOD)
    python_time = time.time() - start
    print(f"  Time: {python_time:.4f} seconds")
    
    # Test 2: Event Loop Performance
    print("\n[Test 2] Event Loop Performance")
    print("-" * 60)
    NUM_BARS = 50000
    print(f"Processing {NUM_BARS:,} bars through event loop...")
    
    bt = Backtester()
    strategy = bt.add_strategy(SimpleStrategy)
    
    # Generate dummy bars
    bars = []
    price = 100.0
    for i in range(NUM_BARS):
        price += (random.random() - 0.5)
        bar = Bar()
        bar.timestamp = i * 60 * 1000000000  # nanoseconds
        bar.instrument_id = 1
        bar.close = price
        bar.open = price
        bar.high = price
        bar.low = price
        bar.volume = 1000
        bars.append(bar)
    
    bt.add_data(bars)
    
    start = time.time()
    bt.run()
    engine_time = time.time() - start
    
    print(f"  Time: {engine_time:.4f} seconds")
    print(f"  Throughput: {NUM_BARS / engine_time:,.0f} bars/second")
    print(f"  Bars processed: {strategy.bar_count:,}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"SMA Calculation ({DATA_SIZE:,} prices): {python_time:.4f}s")
    print(f"Event Loop ({NUM_BARS:,} bars): {engine_time:.4f}s")
    print(f"Event Loop Throughput: {NUM_BARS / engine_time:,.0f} bars/sec")
    
    print("\nNOTE: GPU acceleration unavailable due to CUDA version mismatch")
    print("  Driver: 555.42.06 (supports CUDA 12.5)")
    print("  Runtime: CUDA 13.0")
    print("  To enable GPU: upgrade NVIDIA driver or recompile with CUDA 12.x")
    print("=" * 60)

if __name__ == "__main__":
    main()
