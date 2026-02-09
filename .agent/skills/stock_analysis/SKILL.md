---
name: daily_stock_analysis
description: 自动化港股日常选股、技术分析及报告生成工作流，集成 LongPort 实时数据。
---
# 每日股票分析工作流 (Daily Stock Analysis Workflow)

本 Skill 将引导您完成从自动化选股到深度技术验证及报告生成的全流程。

## 1. 自动化初筛 (Automated Screening)

运行选股脚本生成初始候选名单。

```bash
# // turbo
.venv/bin/python watch_stock/watch_metric.py
```

## 2. 报告读取 (Report Retrieval)

定位并查看生成的特征报告。

- **路径**: `data/features/yyyy-mm-dd.md` (例如 `2026-01-26.md`)
- **操作**: 使用 `view_file` 读取内容。

## 3. 持仓监控与体检 (Monitor Holdings & Portfolio)

在分析新标的之前，优先检查现有持仓的健康状况。

- 获取持仓信息 `mcp_stock_positions` 和 `mcp_fund_positions`
- **穿透分析**: 如果持有的是衍生品 (如涡轮 Warrant、牛熊证 CBBC)，**必须分析其正股 (Underlying Stock)**。衍生品价格是正股价格的函数。
- **衍生品风控 (Critical Risk Check)**: ⚠️ **对于持仓涡轮，必须检查以下红线**：

  1. **到期日 (Maturity)**: 剩余时间若 **< 60天**，时间损耗 (Theta) 加速，**必须强制预警**。
  2. **街货比 (Street Ratio)**: 若 **> 40%**，做市商控盘能力下降，价格容易失真，**必须强制预警**。
  3. **引伸波幅 (IV Trend)**: 防止"缩窝"。若 IV 显著高于同类平均或处于历史高位 (>90% 分位)，**必须预警**。
  4. **买卖价差 (Spread)**: 检查流动性隐形损耗。若价差 > 5-8 ticks (或 > 1.5%)，**建议换仓**。

  - **操作**: 使用 `mcp_static_info` 或 `search_web` 获取数据。
- **验证**: 对正股使用 `mcp_quote` 和 `mcp_candlesticks`，检查原有的交易逻辑 (做多/做空) 是否仍然有效，或是否触发止损/止盈。

## 4. 技术形态识别 (Pattern Recognition)

依托 `pattern_recognition.md` 中定义的 60+ 种技术形态进行深度扫描。

- **引用文件**: `.agent/skills/stock_analysis/pattern_recognition.md`
- **操作**:
  1. 使用 `mcp_candlesticks` 获取近期K线。
  2. 对照模式库识别反转形态，重点关注以下高胜率形态：
     - **顶部反转**: `CDLSHOOTINGSTAR` (射击之星), `CDLDARKCLOUDCOVER` (乌云压顶), `CDLEVENINGSTAR` (暮星)。
     - **底部反转**: `CDLHAMMER` (锤头), `CDLPIERCING` (刺透), `CDLMORNINGSTAR` (晨星)。
     - **中继形态**: `CDLDOJI` (十字星 - 需结合位置判断), `CDLMARUBOZU` (光头光脚 - 趋势增强)。
  3. **验证**: 确认形态出现的位置（是否在布林带轨/支撑阻力位），位置决定有效性。

## 5. 市场深度与环境分析 (LongPort MCP 集成)

针对报告中识别的新候选股，或现有的持仓股，使用 `longport-mcp` 进行以下验证：

### A. 市场温度 (Market Temperature)

- **工具**: `mcp_current_market_temperature`
- **目的**: 评估整体市场情绪是否支持交易方向 (例如：在"寒冷"的市场中避免激进做多)。

### B. 技术与资金流验证 (个股维度)

使用以下工具进行深度分析：

1. **价格与结构**: `mcp_candlesticks` (日线/周线) & `mcp_quote`
   - 检查反转形态 (十字星、锤头、射击之星)。
   - 分析布林带位置 (Bandwidth) 及 KDJ 指标。
2. **机构资金流向**: `mcp_capital_distribution` & `mcp_capital_flow`
   - **做多**: 寻找"特大单 (Super Large)"或"大单 (Large)"的净流入。
   - **做空**: 寻找显著的净流出。
3. **流动性与买卖点**: `mcp_depth` (市场深度/摆盘)
   - 识别"买单墙 (Bid Walls)"作为支撑，或"卖单墙 (Ask Walls)"作为阻力，以精确设定入场点或止损位。

## 6. 消息面与情绪 (News & Sentiment)

- **工具**: `search_web`
- **查询**: `[代码] [名称] 下跌原因` 或 `[代码] [名称] 最新消息`。
- **目的**: 验证技术信号 (例如"利好出尽/Sell the News") 是否有基本面事件支撑。

## 7. 策略制定

7.1  3天持有期

7.2  中长期（3个月及以上）持有

### 建议格式

- **方向**: 看多 (Long) / 看空 (Short) / 观望 (Watch)
- **类型**: 激进 (Aggressive) / 防守 (Defensive) / 成长 (Growth)
- **衍生品建议 (关键)**:
  - **认沽证 (Put Warrant)**: 用于做空策略。
  - **认购证 (Call Warrant)**: 用于做多策略。
  - **参数选择**:
    - **到期日 (Maturity)**: **> 60天** (优选 3-6 个月)，避免末日轮。
    - **街货比 (Street Ratio)**: **< 40%** (避免高街货致价格失真)。
    - **引伸波幅 (IV Check)**: 选择 IV 较低或合理的发行商，拒绝高溢价。
    - **流动性 (Spread)**: 买卖价差 **< 3 ticks** (低成本进出)。
    - **价外程度 (Moneyness)**: 3-5% OTM (价外) 以获取杠杆，或 ITM (价内) 以做防守。
    - **实际杠杆 (Effective Gearing)**: 5-8倍。

## 8. 报告生成 (Report Generation)

将最终分析结果追加写入每日报告文件。

- **文件**: `data/analyze_report/yyyy-mm-ar.md`
- **格式**:
  ```markdown
  ## yyyy-mm-dd 技术分析报告 (收盘终稿)

  ### 1. 市场概况
  ...

  ### 2. 重点标的分析
  #### [代码] 名称 - 策略
  ...

  ### 3. 衍生品策略
  | 标的 | 策略 | 推荐工具 | 核心参数 | 逻辑 |
  | ---- | ---- | -------- | -------- | ---- |
  ...
  ```
