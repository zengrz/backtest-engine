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

def run_benchmark(data_size, period, name):
    """Run a single benchmark test"""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Dataset: {data_size:,} prices, Period: {period}")
    
    # Generate data
    prices = [100.0 + random.random() * 100 for _ in range(data_size)]
    
    bt = Backtester()
    
    # Benchmark CPU
    print("CPU: ", end="", flush=True)
    start_cpu = time.time()
    sma_cpu = calculate_sma_python(prices, period)
    cpu_time = time.time() - start_cpu
    print(f"{cpu_time:.4f}s")
    
    # Benchmark GPU
    print("GPU: ", end="", flush=True)
    start_gpu = time.time()
    sma_gpu = bt.compute_sma_gpu(prices, period)
    gpu_time = time.time() - start_gpu
    print(f"{gpu_time:.4f}s")
    
    # Quick verification (check a few points)
    mismatches = 0
    for i in range(period, min(len(prices), period + 100), 10):
        if abs(sma_cpu[i] - sma_gpu[i]) > 1e-5:
            mismatches += 1
    
    if mismatches == 0:
        print("✓ Verification: PASSED")
    else:
        print(f"✗ Verification: FAILED ({mismatches} mismatches)")
    
    speedup = cpu_time / gpu_time
    print(f"Speedup: {speedup:.2f}x", end="")
    if speedup > 1.0:
        print(" 🚀 GPU FASTER!")
    else:
        print(" (CPU faster)")
    
    return cpu_time, gpu_time, speedup

def main():
    print("\n" + "="*60)
    print("GPU vs CPU BENCHMARK SUITE")
    print("Testing different dataset sizes to find GPU advantage")
    print("="*60)
    
    results = []
    
    # Test 1: Small dataset (baseline - CPU should win)
    cpu_time, gpu_time, speedup = run_benchmark(
        data_size=1_000_000,
        period=50,
        name="Test 1: Small Dataset (1M points)"
    )
    results.append(("1M points", cpu_time, gpu_time, speedup))
    
    # Test 2: Medium dataset
    cpu_time, gpu_time, speedup = run_benchmark(
        data_size=5_000_000,
        period=50,
        name="Test 2: Medium Dataset (5M points)"
    )
    results.append(("5M points", cpu_time, gpu_time, speedup))
    
    # Test 3: Large dataset
    cpu_time, gpu_time, speedup = run_benchmark(
        data_size=10_000_000,
        period=50,
        name="Test 3: Large Dataset (10M points)"
    )
    results.append(("10M points", cpu_time, gpu_time, speedup))
    
    # Test 4: Very large dataset
    cpu_time, gpu_time, speedup = run_benchmark(
        data_size=20_000_000,
        period=50,
        name="Test 4: Very Large Dataset (20M points)"
    )
    results.append(("20M points", cpu_time, gpu_time, speedup))
    
    # Test 5: Larger period (more computation per point)
    cpu_time, gpu_time, speedup = run_benchmark(
        data_size=10_000_000,
        period=200,
        name="Test 5: Large Period (10M points, 200-period SMA)"
    )
    results.append(("10M, period=200", cpu_time, gpu_time, speedup))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Test':<25} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<10}")
    print("-"*60)
    for test_name, cpu_t, gpu_t, sp in results:
        marker = "🚀" if sp > 1.0 else "  "
        print(f"{test_name:<25} {cpu_t:>10.4f}s  {gpu_t:>10.4f}s  {sp:>8.2f}x {marker}")
    
    print("\n" + "="*60)
    best_speedup = max(results, key=lambda x: x[3])
    print(f"Best GPU performance: {best_speedup[0]} with {best_speedup[3]:.2f}x speedup")
    print("="*60)

if __name__ == "__main__":
    main()
