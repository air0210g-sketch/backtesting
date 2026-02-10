"""
定时执行「完整 stock_analysis SKILL」：用 Gemini API + Function Calling 跑 §2–§8，再执行 §9.1、§9.2。

依赖：pip install google-genai；环境变量 GEMINI_API_KEY 或 GOOGLE_API_KEY。
LongPort 凭证与现有脚本一致（mcp_config.json 或环境变量），用于行情类 tool。
使用前请确保当日初筛报告已存在（先跑 watch_metric.py 或 run_daily_stock_analysis.py）。

使用（项目根目录）：
  .venv/bin/python scripts/run_full_skill_gemini.py [yyyy-mm-dd]
  不传日期则使用当日。
"""
import json
import os
import subprocess
import sys
import time
from datetime import date

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

SKILL_PATH = os.path.join(_REPO_ROOT, ".agent", "skills", "stock_analysis", "SKILL.md")
REPORT_DIR = os.path.join(_REPO_ROOT, "watch_stock", "report")
CONFIG_PATH = os.path.expanduser("~/.gemini/antigravity/mcp_config.json")


def _load_longport_credentials():
    if not os.path.exists(CONFIG_PATH):
        return None
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    env = {}
    if "mcpServers" in data and "longport-mcp" in data.get("mcpServers", {}):
        env = data["mcpServers"]["longport-mcp"].get("env", {})
    elif "longport-mcp" in data:
        env = data["longport-mcp"].get("env", data["longport-mcp"])
    ak = env.get("LONGPORT_APP_KEY") or os.environ.get("LONGPORT_APP_KEY")
    sk = env.get("LONGPORT_APP_SECRET") or os.environ.get("LONGPORT_APP_SECRET")
    token = env.get("LONGPORT_ACCESS_TOKEN") or os.environ.get("LONGPORT_ACCESS_TOKEN")
    if not all([ak, sk, token]):
        return None
    return (ak, sk, token)


# ---------- Tool 实现（供 Gemini 通过 function call 调用） ----------

def _tool_read_file(rel_path: str) -> str:
    """读仓库内文件，rel_path 相对项目根，如 watch_stock/report/2026-02-10.md。"""
    path = os.path.join(_REPO_ROOT, rel_path) if not os.path.isabs(rel_path) else rel_path
    path = os.path.normpath(path)
    if not path.startswith(_REPO_ROOT):
        return json.dumps({"error": "路径必须在项目内"})
    if not os.path.isfile(path):
        return json.dumps({"error": f"文件不存在: {rel_path}"})
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.dumps({"content": f.read(), "path": rel_path})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_get_quote(symbol: str) -> str:
    """获取单只标的实时行情。symbol 如 700.HK。"""
    creds = _load_longport_credentials()
    if not creds:
        return json.dumps({"error": "未配置 LongPort 凭证"})
    try:
        from longport.openapi import QuoteContext, Config
        cfg = Config(*creds)
        ctx = QuoteContext(cfg)
        q = ctx.quote(symbol)
        # 转为可序列化结构
        out = {}
        for k in ("symbol", "last_done", "open", "high", "low", "prev_close", "volume", "turnover"):
            if hasattr(q, k):
                out[k] = getattr(q, k)
        return json.dumps(out)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_get_candlesticks(symbol: str, period: str = "day", count: int = 30) -> str:
    """获取 K 线。period: day/week; count 默认 30。"""
    creds = _load_longport_credentials()
    if not creds:
        return json.dumps({"error": "未配置 LongPort 凭证"})
    try:
        from longport.openapi import QuoteContext, Config, Period, AdjustType
        cfg = Config(*creds)
        ctx = QuoteContext(cfg)
        p = Period.Day if period == "day" else Period.Week
        candles = ctx.candlesticks(symbol, p, count, AdjustType.ForwardAdjust)
        rows = []
        for c in (candles or [])[-count:]:
            rows.append({
                "time": getattr(c, "timestamp", None),
                "open": getattr(c, "open", None),
                "high": getattr(c, "high", None),
                "low": getattr(c, "low", None),
                "close": getattr(c, "close", None),
                "volume": getattr(c, "volume", None),
            })
        return json.dumps({"symbol": symbol, "period": period, "candles": rows})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_get_market_temperature(market: str = "HK") -> str:
    """获取市场温度。market: HK/US/CN/SG。"""
    creds = _load_longport_credentials()
    if not creds:
        return json.dumps({"error": "未配置 LongPort 凭证"})
    try:
        from longport.openapi import QuoteContext, Config
        cfg = Config(*creds)
        ctx = QuoteContext(cfg)
        t = ctx.current_market_temperature(market)
        return json.dumps({"market": market, "temperature": getattr(t, "temperature", None)})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_write_analysis_report(date_str: str, markdown_content: str) -> str:
    """将分析报告正文写入 watch_stock/report/yyyy-mm-dd_analysis.md。仅接受日期与 Markdown 字符串。"""
    if len(date_str) != 10 or date_str[4] != "-" or date_str[7] != "-":
        return json.dumps({"error": "date_str 须为 yyyy-mm-dd"})
    path = os.path.join(REPORT_DIR, f"{date_str}_analysis.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        return json.dumps({"ok": True, "path": path})
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOL_IMPLS = {
    "read_file": _tool_read_file,
    "get_quote": _tool_get_quote,
    "get_candlesticks": _tool_get_candlesticks,
    "get_market_temperature": _tool_get_market_temperature,
    "write_analysis_report": _tool_write_analysis_report,
}


def _get_function_declarations():
    """Gemini function declarations（与 TOOL_IMPLS 对应）。"""
    return [
        {
            "name": "read_file",
            "description": "读取项目内文件内容。path 为相对项目根的路径，如 watch_stock/report/2026-02-10.md 或 .agent/skills/stock_analysis/pattern_recognition.md",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "相对项目根的文件路径"}},
                "required": ["path"],
            },
        },
        {
            "name": "get_quote",
            "description": "获取单只股票实时行情。symbol 格式如 700.HK、AAPL.US",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
        {
            "name": "get_candlesticks",
            "description": "获取标的 K 线。period 为 day 或 week，count 为根数",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "period": {"type": "string", "description": "day 或 week", "default": "day"},
                    "count": {"type": "integer", "description": "K 线根数", "default": 30},
                },
                "required": ["symbol"],
            },
        },
        {
            "name": "get_market_temperature",
            "description": "获取市场温度，market 为 HK/US/CN/SG",
            "parameters": {
                "type": "object",
                "properties": {"market": {"type": "string", "default": "HK"}},
                "required": [],
            },
        },
        {
            "name": "write_analysis_report",
            "description": "将当日技术分析报告（完整 Markdown 正文）写入 report/yyyy-mm-dd_analysis.md。在完成 §2–§7 后调用一次，传入日期与完整报告内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_str": {"type": "string", "description": "日期 yyyy-mm-dd"},
                    "markdown_content": {"type": "string", "description": "报告全文 Markdown"},
                },
                "required": ["date_str", "markdown_content"],
            },
        },
    ]


def _run_tool(name: str, args: dict) -> str:
    if name not in TOOL_IMPLS:
        return json.dumps({"error": f"未知工具: {name}"})
    fn = TOOL_IMPLS[name]
    try:
        if name == "read_file":
            return fn(args.get("path", ""))
        if name == "get_quote":
            return fn(args.get("symbol", ""))
        if name == "get_candlesticks":
            return fn(args.get("symbol", ""), args.get("period", "day"), args.get("count", 30))
        if name == "get_market_temperature":
            return fn(args.get("market", "HK"))
        if name == "write_analysis_report":
            return fn(args.get("date_str", ""), args.get("markdown_content", ""))
        return json.dumps({"error": "未实现"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _make_function_response_part(types, name: str, result: str):
    """构造回传给模型的 function response Part，兼容不同版本 google-genai。"""
    response_dict = {"result": result} if isinstance(result, str) else result
    try:
        return types.Part.from_function_response(name=name, response=response_dict)
    except TypeError:
        pass
    try:
        return types.Part.from_function_response(name=name, response=result)
    except Exception:
        pass
    if hasattr(types, "FunctionResponse"):
        return types.Part(function_response=types.FunctionResponse(name=name, response=response_dict))
    raise RuntimeError("当前 google-genai 版本无法构造 function response，请升级或查看 types.Part 文档。")


def main():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("请设置环境变量 GEMINI_API_KEY 或 GOOGLE_API_KEY")
        sys.exit(1)

    date_str = sys.argv[1].strip() if len(sys.argv) >= 2 else date.today().strftime("%Y-%m-%d")
    screening_path = os.path.join(REPORT_DIR, f"{date_str}.md")
    if not os.path.isfile(screening_path):
        print(f"初筛报告不存在: {screening_path}，请先运行 watch_metric.py")
        sys.exit(1)

    with open(SKILL_PATH, "r", encoding="utf-8") as f:
        skill_text = f.read()
    with open(screening_path, "r", encoding="utf-8") as f:
        screening_text = f.read()

    user_prompt = f"""你是一个严格执行「每日股票分析」工作流的助手。请严格按照下面 SKILL 的 §2–§8 执行（§1 初筛已由脚本完成）。

【SKILL 全文】
{skill_text}

【当日初筛报告】({date_str}.md)
{screening_text}

请依次：读取并理解初筛报告 → 按需调用 get_quote / get_candlesticks / get_market_temperature 等获取行情与市场温度 → 结合形态与资金面逻辑完成 §7 策略制定 → 最后**必须**调用一次 write_analysis_report，传入 date_str="{date_str}" 和完整的 Markdown 报告正文（格式见 SKILL §8）。不要省略 write_analysis_report。"""

    try:
        from google import genai
        from google.genai import types
        from google.genai.errors import ClientError
    except ImportError as e:
        if "genai" in str(e):
            print("请安装: pip install google-genai")
        else:
            raise
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    tools = types.Tool(function_declarations=_get_function_declarations())
    config = types.GenerateContentConfig(
        temperature=0.2,
        tools=[tools],
    )

    def generate_with_retry(max_retries=3):
        last_err = None
        for attempt in range(max_retries):
            try:
                return client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=contents,
                    config=config,
                )
            except ClientError as e:
                last_err = e
                code = getattr(e, "status_code", None) or (e.args[0] if e.args else None)
                if code in (429, 503):
                    wait = (2**attempt) + 1
                    print(f"[429/503] 限流或暂时不可用，{wait}s 后重试 ({attempt + 1}/{max_retries})…")
                    time.sleep(w)
                else:
                    raise
        print("重试次数已用尽。配额与限流说明: https://ai.google.dev/gemini-api/docs/rate-limits")
        raise last_err

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])]
    report_written = False
    max_rounds = 30
    for _ in range(max_rounds):
        response = generate_with_retry()
        if not response.candidates:
            print("模型未返回有效候选，结束。")
            break
        parts = list(response.candidates[0].content.parts)
        fc_part = None
        fc_name = None
        fc_args = None
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc is not None:
                fc_part = part
                fc_name = getattr(fc, "name", None) or (fc.get("name") if isinstance(fc, dict) else None)
                fc_args = getattr(fc, "args", None) or (fc.get("args") if isinstance(fc, dict) else {}) or {}
                if not isinstance(fc_args, dict):
                    fc_args = dict(fc_args) if hasattr(fc_args, "items") else {}
                break
        if fc_name is None:
            break
        print(f"[Tool] {fc_name} {fc_args}")
        result = _run_tool(fc_name, fc_args)
        if fc_name == "write_analysis_report":
            report_written = True
        contents.append(types.Content(role="model", parts=parts))
        # 回传 function 执行结果（兼容不同版本 google-genai）
        resp_part = _make_function_response_part(types, fc_name, result)
        contents.append(types.Content(role="user", parts=[resp_part]))

    if not report_written:
        print("未检测到 write_analysis_report 调用，请检查模型输出或重试。")
        sys.exit(1)

    # §9.1 发报告到 Telegram；§9.2 自选股同步
    analysis_path = os.path.join(REPORT_DIR, f"{date_str}_analysis.md")
    if os.path.isfile(analysis_path):
        from watch_stock.send_analysis_to_telegram import send_report
        if send_report(analysis_path):
            print("已发送分析报告到 Telegram。")
    # §9.2 自选股同步（子进程调用，传入报告路径）
    sync_script = os.path.join(_REPO_ROOT, "watch_stock", "sync_watchlist_from_analysis.py")
    try:
        subprocess.run(
            [sys.executable, sync_script, analysis_path],
            cwd=_REPO_ROOT,
            check=False,
            timeout=60,
        )
    except Exception as e:
        print(f"自选股同步跳过: {e}")
    print("完成。")


if __name__ == "__main__":
    main()
