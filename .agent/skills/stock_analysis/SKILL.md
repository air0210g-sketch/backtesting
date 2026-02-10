---
name: daily_stock_analysis
description: 当用户要求进行港股每日选股、技术分析、持仓/衍生品体检、或生成/发送分析报告与自选股同步时使用。包含运行 watch_metric 初筛、读取报告、LongPort MCP 行情/持仓/资金流验证、形态库对照、撰写 yyyy-mm-dd_analysis.md 及 Telegram 推送与「auto watch」分组覆盖的完整步骤与命令。
---
# 每日股票分析工作流 (Daily Stock Analysis Workflow)

## 何时使用本技能

- 用户说「开始分析」「跑一下今日选股」「按 SKILL 做每日分析」等，需要执行完整工作流时。
- 用户要求对港股持仓做穿透体检、衍生品风控（到期日/街货比/IV/价差）时。
- 用户要求将分析报告发到 Telegram 或把重点标的同步到 LongBridge 自选股「auto watch」时。

按需加载本 Skill 后，依下列步骤执行；涉及脚本与 MCP 的步骤需在项目根目录执行相应命令或调用对应工具。

---

## 1. 自动化初筛 (Automated Screening)

运行选股脚本生成初始候选名单。

```bash
# // turbo
.venv/bin/python watch_stock/watch_metric.py
```

## 2. 报告读取 (Report Retrieval)

定位并查看生成的特征报告。

- **初筛报告路径**: `watch_stock/report/yyyy-mm-dd.md`（例如 `2026-02-09.md`）
- **分析报告路径**: `watch_stock/report/yyyy-mm-dd_analysis.md`（技术分析终稿）
- **操作**: 使用 `view_file` 或直接读取上述路径内容。

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

将最终分析结果写入每日报告文件。

- **文件**: `watch_stock/report/yyyy-mm-dd_analysis.md`
- **格式**:
  ```markdown
  ## yyyy-mm-dd 技术分析报告 (收盘终稿)

  ### 1. 市场概况
  ...

  ### 2. 技术形态识别
  ...

  ### 3. 重点标的分析
  #### 3.1 看多 (Long) / 3.2 观望 (Watch) / 3.3 观望/止盈
  | 代码 | 名称 | 类型/说明 | 逻辑 |
  ...

  ### 4. 衍生品策略
  ...
  ```

## 9. 通知与自选股同步 (Notification & Watchlist Sync)

报告生成完成后，可按需执行以下两步，将结果推送并同步到日常使用环境。

### 9.1 发送分析报告到 Telegram

将当日分析报告内容以 HTML 消息形式发送到 Telegram（按二级标题每两章一条，超长自动拆分；解析失败时降级为纯文本）。

```bash
# 需先设置环境变量 TELEGRAM_TOKEN、TELEGRAM_CHAT_ID
.venv/bin/python watch_stock/send_analysis_to_telegram.py [yyyy-mm-dd]
```

- 不传日期则发送当日 `report/yyyy-mm-dd_analysis.md`。
- 若参数为文件路径（如 `watch_stock/report/2025-11-27.html`），则将该文件以**文档**形式发送。

### 9.2 同步重点标的自选股分组 (LongBridge)

将报告中 **「## 3. 重点标的分析」** 表格内的股票代码，覆盖到 LongBridge 自选股分组 **「auto watch」**，便于在 App/行情端统一查看。

```bash
# 依赖 longport，凭证同 download_candles（mcp_config.json 或环境变量）
.venv/bin/python watch_stock/sync_watchlist_from_analysis.py [report_path]
```

- 不传参数则使用当日 `watch_stock/report/yyyy-mm-dd_analysis.md`。
- 若已存在「auto watch」分组则**整体替换**为该报告 §3 的标的；若不存在则**创建**该分组并写入。

**建议顺序**：先完成 §8 报告生成，再执行 §9.1（Telegram）、§9.2（自选股同步）。

---

## 脚本与资源索引 (Level 3 按需加载)

| 用途 | 路径 | 说明 |
|------|------|------|
| 初筛 | `watch_stock/watch_metric.py` | 生成 `report/yyyy-mm-dd.md` 候选名单 |
| 报告推送 | `watch_stock/send_analysis_to_telegram.py` | 发送 `yyyy-mm-dd_analysis.md` 到 Telegram |
| 自选股同步 | `watch_stock/sync_watchlist_from_analysis.py` | 用报告 §3 表格覆盖 LongBridge「auto watch」 |
| 形态库 | `.agent/skills/stock_analysis/pattern_recognition.md` | 60+ 技术形态定义，§4 形态识别时引用 |

报告路径约定：初筛 `watch_stock/report/yyyy-mm-dd.md`，终稿 `watch_stock/report/yyyy-mm-dd_analysis.md`。MCP 使用 longport-mcp（行情、持仓、资金流、深度等）及 search_web；本 Skill 定义调用顺序与业务逻辑，不替代 MCP 连接。
