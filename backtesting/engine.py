from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    timestamp: Optional[datetime] = None

@dataclass
class Fill:
    order: Order
    fill_price: Decimal
    fill_quantity: Decimal
    commission: Decimal
    slippage: Decimal
    timestamp: datetime

@dataclass
class Position:
    symbol: str
    quantity: Decimal = Decimal("0")
    avg_cost: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    def update(self, fill: Fill) -> None:
        if fill.order.side == OrderSide.BUY:
            new_quantity = self.quantity + fill.fill_quantity
            if new_quantity != 0:
                # Update weighted average cost
                total_cost = self.quantity * self.avg_cost + fill.fill_quantity * fill.fill_price
                self.avg_cost = total_cost / new_quantity
            self.quantity = new_quantity
        else:
            # Sell
            # PnL = (Sell Price - Avg Cost) * Quantity
            self.realized_pnl += fill.fill_quantity * (fill.fill_price - self.avg_cost)
            self.quantity -= fill.fill_quantity

@dataclass
class Portfolio:
    cash: Decimal
    positions: Dict[str, Position] = field(default_factory=dict)
    initial_capital: Decimal = Decimal("0") # Track initial for proper return calc

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def process_fill(self, fill: Fill) -> None:
        position = self.get_position(fill.order.symbol)
        position.update(fill)

        total_cost = fill.fill_price * fill.fill_quantity
        if fill.order.side == OrderSide.BUY:
            self.cash -= total_cost + fill.commission
        else:
            self.cash += total_cost - fill.commission

    def get_equity(self, prices: Dict[str, Decimal]) -> Decimal:
        equity = self.cash
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                price = prices.get(symbol)
                if price is not None:
                    equity += position.quantity * price
        return equity

class Strategy(ABC):
    @abstractmethod
    def on_bar(self, timestamp: datetime, data: Dict[str, pd.Series]) -> List[Order]:
        """
        data: Dict where key is symbol, value is the row (Series) for that timestamp.
        NOTE: This is a slight deviation from Playbook pattern to handle Multi-Symbol easily.
        """
        pass

    @abstractmethod
    def on_fill(self, fill: Fill) -> None:
        pass

class ExecutionModel(ABC):
    @abstractmethod
    def execute(self, order: Order, bar: pd.Series) -> Optional[Fill]:
        pass

class SimpleExecutionModel(ExecutionModel):
    def __init__(self, slippage_bps: float = 0, commission_rate: float = 0.0003):
        self.slippage_bps = slippage_bps
        self.commission_rate = commission_rate

    def execute(self, order: Order, bar: pd.Series) -> Optional[Fill]:
        # Simple assumption: Fill at Open of the NEXT bar (Simulated here by passing the CURRENT bar?)
        # Wait, usually strategy produces signal at Close of Bar T.
        # Execution happens at Open of Bar T+1.
        # But here 'bar' is passed by Backtester.
        # If Backtester executes pending orders using CURRENT bar, it means we are filling at current bar prices.
        # If strategy logic looked at T-1, and we execute at T (Open/Close), that's fine.
        
        # Let's assume we fill at Market (Open or Close). 
        # For simplicity, fill at 'open' of the current bar passed in.
        
        if order.order_type == OrderType.MARKET:
            try:
                base_price = Decimal(str(bar["open"]))
            except KeyError:
                base_price = Decimal(str(bar["close"])) # Fallback

            # Apply slippage
            slippage_mult = Decimal("1") + (Decimal(str(self.slippage_bps)) / Decimal("10000"))
            if order.side == OrderSide.BUY:
                fill_price = base_price * slippage_mult
            else:
                fill_price = base_price / slippage_mult

            # Commission
            commission = fill_price * order.quantity * Decimal(str(self.commission_rate))
            slippage = abs(fill_price - base_price) * order.quantity

            return Fill(
                order=order,
                fill_price=fill_price,
                fill_quantity=order.quantity,
                commission=commission,
                slippage=slippage,
                timestamp=bar.name
            )
        return None

class Backtester:
    def __init__(
        self,
        strategy: Strategy,
        execution_model: ExecutionModel,
        initial_capital: Decimal = Decimal("100000")
    ):
        self.strategy = strategy
        self.execution_model = execution_model
        self.portfolio = Portfolio(cash=initial_capital, initial_capital=initial_capital)
        self.equity_curve: List[tuple] = []
        self.trades: List[Fill] = []
        self.strategy.broker = self # Allow strategy to access portfolio/broker

    def run(self, data: Dict[str, pd.DataFrame]):
        """
        Run backtest.
        data: Dict[str, DataFrame] -> Key: Symbol, DF: OHLCV with Datetime Index
        """
        # 1. Align data indices (Union of all timestamps)
        all_timestamps = sorted(list(set().union(*[df.index for df in data.values()])))
        
        pending_orders: List[Order] = []

        print(f"Starting backtest on {len(all_timestamps)} bars...")

        for timestamp in all_timestamps:
            # current_bars: Dict[symbol, Series] for this timestamp
            current_bars = {}
            current_prices = {} 

            for symbol, df in data.items():
                if timestamp in df.index:
                    bar = df.loc[timestamp]
                    current_bars[symbol] = bar
                    current_prices[symbol] = Decimal(str(bar["close"]))
            
            if not current_bars:
                continue

            # 2. Execute pending orders (from previous step) at Today's prices (Open)
            # Filter orders for symbols that have data today
            next_pending = []
            for order in pending_orders:
                if order.symbol in current_bars:
                    bar = current_bars[order.symbol]
                    fill = self.execution_model.execute(order, bar)
                    if fill:
                        self.portfolio.process_fill(fill)
                        self.strategy.on_fill(fill)
                        self.trades.append(fill)
                    else:
                        pass # Valid order but not filled (e.g. limit)
                else:
                    next_pending.append(order) # Keep pending if no data
            
            pending_orders = next_pending

            # 3. Update Equity Curve
            equity = self.portfolio.get_equity(current_prices)
            self.equity_curve.append((timestamp, float(equity)))

            # 4. Strategy generates NEW orders (based on Today's Close data)
            # Pass all available history up to now? Or just current bar?
            # Strategy needs history. We can pass the full DF but Strategy should only look at loc[:timestamp]
            # Ideally strategy caches or we pass a slice.
            # For efficiency in Python, passing full DF and current timestamp is better.
            
            # NOTE: Strategy inside needs to be careful not to look ahead.
            new_orders = self.strategy.on_bar(timestamp, current_bars)
            pending_orders.extend(new_orders)

        return self._create_results()

    def _create_results(self) -> pd.DataFrame:
        if not self.equity_curve:
            return pd.DataFrame()
        equity_df = pd.DataFrame(self.equity_curve, columns=["timestamp", "equity"])
        equity_df.set_index("timestamp", inplace=True)
        equity_df["returns"] = equity_df["equity"].pct_change().fillna(0)
        return equity_df
