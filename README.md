# vbt_backtesting

基于 **vectorbt** 的量化回测框架，并集成港股每日选股、技术分析工作流与 LongPort 数据/自选股能力。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- **TA-Lib**：形态与 KDJ 等指标依赖 TA-Lib，需先安装 C 库（macOS: `brew install ta-lib`），再 `pip install TA-Lib`。
- **可选**：数据下载与 LongPort 相关脚本需 `pip install longport-openapi`，凭证见下方「配置」。

## 项目结构

| 目录/文件 | 说明 |
|-----------|------|
| **`backtesting/`** | 回测核心：`framework.py`（VectorBT 封装）、`indicators.py`、`data_loader.py`、`notifier.py`（Telegram） |
| **`watch_stock/`** | 每日选股与报告：初筛、报告生成、Telegram 推送、自选股同步 |
| **`scripts/`** | 策略执行、数据工具、定时任务等脚本 |
| **`.agent/skills/`** | Agent 技能：stock_analysis、longport-fetcher 等（见 SKILL 描述） |
| **`stock_data/`** | 行情 CSV、`stock_list.json`、`stock_names.json` |
| **`watch_stock/report/`** | 初筛报告 `yyyy-mm-dd.md`、分析终稿 `yyyy-mm-dd_analysis.md` |

更多架构说明见 **`project_summary.md`**。

## 回测框架 (backtesting/)

- **策略入口**：`scripts/run_struct.py` — 定义策略、参数网格、训练/验证拆分与回测。
- **指标与数据**：`indicators.py` 提供 KDJ/ATR 等；`data_loader.py` 从 `stock_data/` 读 CSV 并对齐。
- **通知**：`notifier.py` 支持 `send_message` / `send_document` / `send_photo`，用于 Telegram 推送。

## 每日选股工作流 (watch_stock/)

完整流程见 **`.agent/skills/stock_analysis/SKILL.md`**，可自动化部分如下：

| 步骤 | 命令/脚本 | 说明 |
|------|-----------|------|
| 初筛 | `watch_stock/watch_metric.py` | 从 `stock_list.json` 读标的，按 KDJ 金叉/周线 J&lt;5 等筛选，写 `report/yyyy-mm-dd.md`，可选 Telegram 摘要 |
| 发报告 | `watch_stock/send_analysis_to_telegram.py [yyyy-mm-dd]` | 将 `yyyy-mm-dd_analysis.md` 以**文档**形式发到 Telegram |
| 自选股 | `watch_stock/sync_watchlist_from_analysis.py [report_path]` | 将报告 §3 表格中的代码覆盖到 LongBridge「auto watch」分组 |

**定时任务**（UTC+8 每个交易日 11:30、15:30、17:00 执行初筛）：

```bash
.venv/bin/python scripts/run_daily_stock_analysis.py
```

## 数据与脚本

- **数据下载/更新/校验**：见 **`scripts/data_tools/README.md`**。数据落盘到 `stock_data/`，命名如 `{symbol}_day.csv`。
- **Skill 内推荐下载方式**：`.agent/skills/longport-fetcher/scripts/download_candles.py`，支持 watchlist、增量、默认「观察」「esg 50」。

其他脚本示例：`run_struct.py`（回测）、`test_features.py`（Telegram/定时测试）、`analyze_kdj_frequency.py`（指标分析）。

## Agent 技能 (.agent/skills / .agents/skills)

| 技能 | 用途 |
|------|------|
| **stock_analysis** | 港股每日选股、技术分析、持仓/衍生品体检、报告生成与推送、自选股同步；含形态库 `pattern_recognition.md` |
| **longport-fetcher** | 按 watchlist 或单标的拉取行情并落盘到 `stock_data/`，含 `download_candles.py` 与路径约定 |
| **backtesting-frameworks** | 回测规范：防 look-ahead、幸存者偏差、成本与 PIT 数据、walk-forward 等，见 `resources/` |

## 配置

- **Telegram**：环境变量 `TELEGRAM_TOKEN`、`TELEGRAM_CHAT_ID`，用于选股报告推送与回测通知。
- **LongPort**：`mcp_config.json`（如 `~/.gemini/antigravity/mcp_config.json`）中 `longport-mcp` 的 `env`：`LONGPORT_APP_KEY`、`LONGPORT_APP_SECRET`、`LONGPORT_ACCESS_TOKEN`；或直接设环境变量。
- **选股标的**：`stock_data/stock_list.json` 为代码列表；名称由 `watch_metric.py` 通过 LongPort static_info 拉取并写入 `stock_data/stock_names.json`。

## 依赖概览

见 **`requirements.txt`**。核心：pandas、numpy、vectorbt、TA-Lib、python-telegram-bot、schedule、plotly；数据相关可选 longport-openapi。
