import time
import schedule
from backtesting.framework import BacktestRunner
from backtesting.notifier import TelegramNotifier
import os


def mock_job(runner):
    print("⏰ 任务触发！")
    runner.notify("⏰ 计划任务已触发！")


def test_telegram_features():
    print("--- 测试 Telegram 功能 ---")
    token = os.environ.get("TELEGRAM_TOKEN", "FAKE_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "FAKE_CHAT_ID")

    print(f"使用 Token: {token[:4]}*** | Chat ID: {chat_id}")  # Masked output

    runner = BacktestRunner(
        data_dir="stock_data", telegram_token=token, telegram_chat_id=chat_id
    )

    # 1. Test Direct Notification
    print("尝试发送启动通知...")
    runner.notify("🚀 测试脚本已启动")

    # 2. Test Scheduling
    print("计划每 2 秒运行一次模拟任务...")
    runner.run_scheduled_job(mock_job, "seconds", 2, runner=runner)


if __name__ == "__main__":
    try:
        test_telegram_features()
    except KeyboardInterrupt:
        print("用户已停止。")
