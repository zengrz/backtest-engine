import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from backtester import Backtester, Strategy, Bar, Side, Order, OrderType
    print("Successfully imported Backtester, Strategy, Bar, Side, Order, OrderType")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
