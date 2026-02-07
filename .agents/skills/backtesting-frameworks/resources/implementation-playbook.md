# Backtesting Frameworks Implementation Playbook

This file contains detailed patterns, checklists, and code samples referenced by the skill.

## Core Concepts

### 1. Backtesting Biases

| Bias             | Description               | Mitigation              |
| ---------------- | ------------------------- | ----------------------- |
| **Look-ahead**   | Using future information  | Point-in-time data      |
| **Survivorship** | Only testing on survivors | Use delisted securities |
| **Overfitting**  | Curve-fitting to history  | Out-of-sample testing   |
| **Selection**    | Cherry-picking strategies | Pre-registration        |
| **Transaction**  | Ignoring trading costs    | Realistic cost models   |

### 2. Proper Backtest Structure

```
Historical Data
      │
      ▼
┌─────────────────────────────────────────┐
│              Training Set               │
│  (Strategy Development & Optimization)  │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│             Validation Set              │
│  (Parameter Selection, No Peeking)      │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│               Test Set                  │
│  (Final Performance Evaluation)         │
└─────────────────────────────────────────┘
```

### 3. Walk-Forward Analysis

```
Window 1: [Train──────][Test]
Window 2:     [Train──────][Test]
Window 3:         [Train──────][Test]
Window 4:             [Train──────][Test]
                                     ─────▶ Time
```

## Implementation Patterns
## Key Capabilities (VectorBT)
- https://github.com/ChadThackray/vectorbt-for-beginners-2022
When using `vectorbt`, leverage these core features for high-performance research:

1. **Broadcasting (The Nuclear Option)**
    - Run backtests on **M params x N stocks** simultaneously without loops.
    - Example: `vbt.MACD.run(close, window=[10, 20, 30])` broadens inputs to test all window sizes at once.

2. **Indicator Factory**
    - Wrap custom logic using `vbt.IndicatorFactory` to enable auto-broadcasting and grid search.
    - Avoid manual Pandas loops; let the factory handle parameter expansion.

3. **Advanced Reality Checks**
    - Use `from_signals` built-in order types for realistic simulation:
        - `sl_stop`: Stop Loss
        - `tp_stop`: Take Profit
        - `ts_stop`: Trailing Stop

4. **Parameter Optimization**
    - Use broadcasting to run thousands of simulations (Grid Search).
    - Analyze results using Heatmaps (`vbt.heatmap`) to find robust parameter stables, distinguishing signal from noise.


## Best Practices

### Do's
- **Use point-in-time data** - Avoid look-ahead bias
- **Include transaction costs** - Realistic estimates
- **Test out-of-sample** - Always reserve data
- **Use walk-forward** - Not just train/test
- **Monte Carlo analysis** - Understand uncertainty
- **Output must include Key Metrics** - Key metrics to monitor in backtesting results include: win rate, profit-to-loss ratio, total number of trades, and maximum drawdown.

### Don'ts
- **Don't overfit** - Limit parameters
- **Don't ignore survivorship** - Include delisted
- **Don't use adjusted data carelessly** - Understand adjustments
- **Don't optimize on full history** - Reserve test set
- **Don't ignore capacity** - Market impact matters

## Resources

- [Advances in Financial Machine Learning (Marcos López de Prado)](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089)
- [Quantitative Trading (Ernest Chan)](https://www.amazon.com/Quantitative-Trading-Build-Algorithmic-Business/dp/1119800064)
- [Backtrader Documentation](https://www.backtrader.com/docu/)
