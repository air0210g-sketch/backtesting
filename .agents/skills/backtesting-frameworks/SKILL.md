---
name: backtesting-frameworks
description: Use when the user is developing trading strategy backtests, validating strategy robustness, or building backtesting infrastructure. Covers look-ahead bias, survivorship bias, transaction costs, point-in-time data, and walk-forward testing. Contains stepwise instructions and references to implementation-playbook and strategy resources; load those files when detailed patterns or examples are needed.
---

# Backtesting Frameworks

Build robust, production-grade backtesting systems that avoid common pitfalls and produce reliable strategy performance estimates.

## When to use this skill

- Developing trading strategy backtests or building backtesting infrastructure.
- Validating strategy performance and robustness (train/validation/test, walk-forward).
- Avoiding common backtesting biases (look-ahead, survivorship, costs).
- Need a clear process: hypothesis → universe → timeframe → evaluation → PIT pipelines → execution logic.

## Do not use this skill when

- You need live trading execution or investment advice.
- Historical data quality is unknown or incomplete.
- The task is only a quick performance summary with no rigor requirement.

## Instructions

- Define hypothesis, universe, timeframe, and evaluation criteria.
- Build point-in-time data pipelines and realistic cost models.
- Implement event-driven simulation and execution logic.
- Use train/validation/test splits and walk-forward testing.
- For detailed patterns and code examples, load `resources/implementation-playbook.md` (Level 3).

## Safety

- Do not present backtests as guarantees of future performance.
- Avoid providing financial or investment advice.

---

## Scripts & resources index (Level 3, load when needed)

| Purpose | Path | Notes |
|---------|------|--------|
| Implementation patterns & examples | `resources/implementation-playbook.md` | Detailed patterns and code examples |
| Strategy design reference | `resources/strategy.md` | Strategy-related reference material |

This skill defines the backtesting process and when to load which resource; use it together with data/execution tools (e.g. MCP or local scripts) as needed.

