import sys
import os
import time
import random

# Add python directory to path
sys.path.append('../python')

try:
    from backtester import Backtester
    print("Successfully imported Backtester")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def calculate_sma_cpu(prices, period):
    sma_values = []
    for i in range(len(prices)):
        if i < period - 1:
            sma_values.append(0.0)
            continue
        
        window = prices[i - period + 1 : i + 1]
        sma = sum(window) / period
        sma_values.append(sma)
    return sma_values

def main():
    # Generate data
    DATA_SIZE = 1000000
    PERIOD = 50
    print(f"Generating {DATA_SIZE} random prices...")
    prices = [100.0 + random.random() * 100 for _ in range(DATA_SIZE)]
    
    bt = Backtester()
    
    # Benchmark CPU
    print("Benchmarking CPU implementation...")
    start_cpu = time.time()
    sma_cpu = calculate_sma_cpu(prices, PERIOD)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu
    print(f"CPU Time: {cpu_time:.4f} seconds")
    
    # Benchmark GPU
    print("Benchmarking GPU implementation...")
    start_gpu = time.time()
    sma_gpu = bt.compute_sma_gpu(prices, PERIOD)
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu
    print(f"GPU Time: {gpu_time:.4f} seconds")
    
    # Verify results (check a few random points)
    print("Verifying results...")
    mismatches = 0
    for i in range(PERIOD, len(prices), 1000): # Check every 1000th point
        if abs(sma_cpu[i] - sma_gpu[i]) > 1e-5:
            mismatches += 1
            if mismatches < 5:
                print(f"Mismatch at index {i}: CPU={sma_cpu[i]}, GPU={sma_gpu[i]}")
    
    if mismatches == 0:
        print("Verification Successful: Results match!")
    else:
        print(f"Verification Failed: {mismatches} mismatches found.")

    print("-" * 30)
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")

if __name__ == "__main__":
    main()
