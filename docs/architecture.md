# Architecture

## Overview
The Backtest Engine is a hybrid C++/Python system designed for high-performance historical simulation.

### Components

1. **Core (C++)**:
   - `EventLoop`: Manages the simulation time and event distribution.
   - `Portfolio`: Tracks cash, positions, and equity.
   - `ExecutionEngine`: Simulates order fills based on market data.

2. **Compute (CUDA)**:
   - `GpuExecution`: Provides GPU-accelerated functions (e.g., SMA, Pricing) via CUDA kernels.

3. **Bindings (pybind11)**:
   - Exposes C++ classes to Python.
   - Allows strategies to be written in Python while the heavy lifting is done in C++.

### Data Flow
1. User loads `Bar` data in Python.
2. Data is pushed to C++ `EventLoop`.
3. `EventLoop` processes bars sequentially.
4. `Strategy.on_bar()` is called (in Python).
5. Strategy submits `Order`s.
6. `ExecutionEngine` fills orders.
7. `Portfolio` updates positions.
