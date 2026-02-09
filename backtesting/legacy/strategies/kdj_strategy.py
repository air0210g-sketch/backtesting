from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Deque
import pandas as pd
import numpy as np
from collections import deque

from backtesting.engine import Strategy, Order, OrderSide, OrderType, Fill

class KDJStrategy(Strategy):
    def __init__(self, n: int = 9, k_period: int = 3, d_period: int = 3, buy_qty: int = 100):
        self.n = n
        self.k_period = k_period
        self.d_period = d_period
        self.buy_qty = Decimal(str(buy_qty))
        
        # History buffer for calculation. Needs at least N periods.
        # Storing DataFrames or dicts? Dictionary of Deques
        self.history: Dict[str, List[pd.Series]] = {}
        
        # Previous K, D values for crossover check
        # Key: Symbol
        self.prev_k: Dict[str, float] = {}
        self.prev_d: Dict[str, float] = {}

    def calculate_kdj(self, symbol: str) -> tuple:
        # Need history dataframe
        data_list = self.history.get(symbol, [])
        if len(data_list) < self.n:
             return 50.0, 50.0, 50.0 # Default neutral
             
        # Create small DF for vector calculation
        # We only really need the last N+few rows to update rolling calc, 
        # But standard KDJ is recursive (EMA-like).
        # Actually standard KDJ uses SMA for K and D smoothing? 
        # K = 2/3 PrevK + 1/3 RSV. This is EMA with alpha=1/3.
        # So we just need previous K and current RSV.
        
        # 1. Calculate Current RSV
        # RSV = (Close - Low_N) / (High_N - Low_N) * 100
        
        # Get last N bars
        recent_bars = data_list[-self.n:]
        highs = [b['high'] for b in recent_bars]
        lows = [b['low'] for b in recent_bars]
        current_close = recent_bars[-1]['close']
        
        highest_high = max(highs)
        lowest_low = min(lows)
        
        if highest_high == lowest_low:
             rsv = 50.0
        else:
             rsv = (current_close - lowest_low) / (highest_high - lowest_low) * 100
             
        # 2. Update K, D
        # Start with 50 if no prev
        prev_k = self.prev_k.get(symbol, 50.0)
        prev_d = self.prev_d.get(symbol, 50.0)
        
        k = (2/3) * prev_k + (1/3) * rsv
        d = (2/3) * prev_d + (1/3) * k
        j = 3 * k - 2 * d
        
        return k, d, j

    def on_bar(self, timestamp: datetime, data: Dict[str, pd.Series]) -> List[Order]:
        orders = []
        
        for symbol, bar in data.items():
            # 1. Update History
            if symbol not in self.history:
                self.history[symbol] = []
            
            # Append current bar (Close is known)
            self.history[symbol].append(bar)
            
            # Avoid unbounded growth (keep e.g. 100 is enough for KDJ if recursive, 
            # but strict recursive needs infinite, practically 50 is fine)
            if len(self.history[symbol]) > 100:
                self.history[symbol].pop(0)
            
            # 2. Calculate Indicators
            curr_k, curr_d, curr_j = self.calculate_kdj(symbol)
            
            # 3. Check Signal (vs Previous)
            prev_k = self.prev_k.get(symbol, 50.0)
            prev_d = self.prev_d.get(symbol, 50.0)
            
            # Golden Cross: K crosses above D
            # Cross: Prev K < Prev D  AND  Curr K > Curr D
            is_gold_cross = (prev_k < prev_d) and (curr_k > curr_d)
            
            # Death Cross: K crosses below D
            is_death_cross = (prev_k > prev_d) and (curr_k < curr_d)
            
            # User Conditions:
            # Buy: Golden Cross AND J < 30
            # Sell: Death Cross AND J > 90
            
            position = self.broker.portfolio.get_position(symbol)
            curr_qty = position.quantity
            
            # Use J of current or previous? "J值" usually means current J at cross point.
            
            if is_gold_cross and curr_j < 30:
                # Signal Buy
                if curr_qty == 0: # Only open if flat? Or accumulate? Let's assume simple: max 1 pos
                    orders.append(Order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=self.buy_qty,
                        order_type=OrderType.MARKET,
                        timestamp=timestamp
                    ))
            
            elif is_death_cross and curr_j > 90:
                # Signal Sell
                if curr_qty > 0:
                    orders.append(Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=curr_qty, # Close all
                        order_type=OrderType.MARKET,
                        timestamp=timestamp
                    ))
            
            # 4. Save state
            self.prev_k[symbol] = curr_k
            self.prev_d[symbol] = curr_d
            
        return orders

    def on_fill(self, fill: Fill) -> None:
        pass
