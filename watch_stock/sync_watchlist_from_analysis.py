"""
从 SKILL 分析报告（*_analysis.md）的「## 3. 重点标的分析」中解析股票代码，
并调用 LongBridge OpenAPI 将这批标的覆盖到「auto watch」自选股分组。

依赖：pip install longport
凭证：与 scripts/data_tools/download_candles.py 相同，从 mcp_config.json 或环境变量读取。

使用（项目根目录）：
  .venv/bin/python watch_stock/sync_watchlist_from_analysis.py [report_path]
  不传参数则使用当日 report/yyyy-mm-dd_analysis.md
"""
import json
import os
import re
import sys
from datetime import date

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(REPO_ROOT, "watch_stock", "report")
CONFIG_PATH = os.path.expanduser("~/.gemini/antigravity/mcp_config.json")
WATCHLIST_GROUP_NAME = "auto watch"

# 表格行中第一列代码的正则（如 9868.HK, 00788.HK, 82333.HK）
SYMBOL_IN_TABLE_RE = re.compile(r"^\|\s*([0-9A-Z]+\.(?:HK|US|SH|SZ))\s*\|", re.MULTILINE)


def get_credentials(config_path=CONFIG_PATH):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    env = {}
    if "mcpServers" in data and "longport-mcp" in data.get("mcpServers", {}):
        env = data["mcpServers"]["longport-mcp"].get("env", {})
    elif "longport-mcp" in data:
        env = data["longport-mcp"].get("env", data["longport-mcp"])
    app_key = env.get("LONGPORT_APP_KEY") or os.environ.get("LONGPORT_APP_KEY")
    app_secret = env.get("LONGPORT_APP_SECRET") or os.environ.get("LONGPORT_APP_SECRET")
    access_token = env.get("LONGPORT_ACCESS_TOKEN") or os.environ.get("LONGPORT_ACCESS_TOKEN")
    if not all([app_key, app_secret, access_token]):
        raise ValueError("缺少 LONGPORT_APP_KEY / LONGPORT_APP_SECRET / LONGPORT_ACCESS_TOKEN")
    return app_key, app_secret, access_token


def extract_symbols_from_section3(md_content: str) -> list[str]:
    """从报告正文中截取「## 3. 重点标的分析」到「## 4.」之间内容，解析表格中的代码列。"""
    start_marker = "## 3. 重点标的分析"
    end_marker = "## 4."
    start = md_content.find(start_marker)
    if start == -1:
        return []
    start += len(start_marker)
    end = md_content.find(end_marker, start)
    if end == -1:
        block = md_content[start:]
    else:
        block = md_content[start:end]
    symbols = []
    for m in SYMBOL_IN_TABLE_RE.finditer(block):
        symbols.append(m.group(1))
    return list(dict.fromkeys(symbols))  # 去重且保持顺序


def main():
    try:
        from longport.openapi import QuoteContext, Config, SecuritiesUpdateMode
    except ImportError:
        print("请先安装: pip install longport")
        sys.exit(1)

    if len(sys.argv) >= 2:
        report_path = sys.argv[1].strip()
        if not os.path.isabs(report_path):
            report_path = os.path.join(REPO_ROOT, report_path)
    else:
        date_str = date.today().strftime("%Y-%m-%d")
        report_path = os.path.join(REPORT_DIR, f"{date_str}_analysis.md")

    if not os.path.isfile(report_path):
        print(f"报告不存在: {report_path}")
        sys.exit(1)

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    symbols = extract_symbols_from_section3(content)
    if not symbols:
        print("未在「## 3. 重点标的分析」中解析到任何股票代码。")
        sys.exit(1)
    print(f"从报告解析到 {len(symbols)} 只标的: {symbols}")

    app_key, app_secret, access_token = get_credentials()
    os.environ["LONGPORT_APP_KEY"] = app_key
    os.environ["LONGPORT_APP_SECRET"] = app_secret
    os.environ["LONGPORT_ACCESS_TOKEN"] = access_token

    cfg = Config.from_env()
    ctx = QuoteContext(cfg)
    groups = ctx.watchlist()
    if not isinstance(groups, list):
        groups = getattr(groups, "groups", groups) or []
    group_id = None
    for g in groups:
        name = getattr(g, "name", None) or (g.get("name") if isinstance(g, dict) else None) or ""
        if name.strip().lower() == WATCHLIST_GROUP_NAME.lower():
            group_id = getattr(g, "id", None) or (g.get("id") if isinstance(g, dict) else None)
            break

    if group_id is not None:
        ctx.update_watchlist_group(
            group_id,
            securities=symbols,
            mode=SecuritiesUpdateMode.Replace,
        )
        print(f"已更新分组「{WATCHLIST_GROUP_NAME}」(id={group_id})，共 {len(symbols)} 只标的。")
    else:
        new_id = ctx.create_watchlist_group(name=WATCHLIST_GROUP_NAME, securities=symbols)
        print(f"已创建分组「{WATCHLIST_GROUP_NAME}」(id={new_id})，共 {len(symbols)} 只标的。")


if __name__ == "__main__":
    main()
