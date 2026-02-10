"""
每个交易日的 11:30、15:30、17:00（UTC+8）自动执行「每日股票分析」工作流（参考 .agent/skills/stock_analysis/SKILL.md）。

执行内容：运行 watch_stock/watch_metric.py（初筛 + 写 report/yyyy-mm-dd.md + 可选 Telegram 通知）。
交易日判断：默认按周一至周五；若需排除港/沪休市日，可后续接入交易日历。

使用（项目根目录）：
  .venv/bin/python scripts/run_daily_stock_analysis.py

依赖：与 test_features.py 相同（backtesting.notifier）；运行 watch_metric 需 stock_data 与 longport（可选）。需 Python 3.9+（zoneinfo）。
"""
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

# 时区：UTC+8（与港/沪一致）
TZ_UTC8 = ZoneInfo("Asia/Shanghai")
# 触发时刻（UTC+8 的 HH:MM）
TARGET_TIMES = ("11:30", "15:30", "17:00")

# 项目根目录，保证可导入 watch_stock、backtesting
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def is_trading_day() -> bool:
    """当前是否为交易日（按 UTC+8 的周一至周五）。"""
    now = datetime.now(TZ_UTC8)
    return now.weekday() < 5  # 0=Mon, 4=Fri


def run_stock_analysis_job():
    """执行 SKILL 第一步：初筛并写报告。仅交易日执行。"""
    if not is_trading_day():
        return
    now = datetime.now(TZ_UTC8).strftime("%Y-%m-%d %H:%M")
    print(f"[{now} UTC+8] 触发每日选股分析（watch_metric）...")
    try:
        from watch_stock.watch_metric import run
        run()
        # 可选：发送完成提醒（需 TELEGRAM_TOKEN / TELEGRAM_CHAT_ID）
        token = os.environ.get("TELEGRAM_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if token and chat_id:
            from backtesting.notifier import TelegramNotifier
            TelegramNotifier(token=token, chat_id=chat_id).send_message(
                f"📊 定时选股已执行 {now}\n路径: watch_stock/report/"
            )
    except Exception as e:
        print(f"[run_daily_stock_analysis] 执行失败: {e}")
        if os.environ.get("TELEGRAM_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
            try:
                from backtesting.notifier import TelegramNotifier
                TelegramNotifier(
                    token=os.environ.get("TELEGRAM_TOKEN"),
                    chat_id=os.environ.get("TELEGRAM_CHAT_ID"),
                ).send_message(f"⚠️ 定时选股失败 {now}: {e}")
            except Exception:
                pass


def main():
    print("每日选股定时任务已启动：11:30 / 15:30 / 17:00（UTC+8，仅交易日执行）")
    print("按 Ctrl+C 停止。")
    last_triggered = set()  # (date_str, slot) 已触发，避免同一分钟重复
    while True:
        now = datetime.now(TZ_UTC8)
        date_str = now.strftime("%Y-%m-%d")
        slot = now.strftime("%H:%M")
        key = (date_str, slot)
        if slot in TARGET_TIMES and key not in last_triggered and is_trading_day():
            run_stock_analysis_job()
            last_triggered.add(key)
        # 新的一天清空，避免 set 无限增大
        if len(last_triggered) > 10:
            last_triggered = {k for k in last_triggered if k[0] == date_str}
        time.sleep(30)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("用户已停止。")
