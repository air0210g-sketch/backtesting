"""
将当日分析报告以 md 文档形式直接发送到 Telegram。

依赖：与 backtesting.notifier 相同（python-telegram-bot）；需设置 TELEGRAM_TOKEN、TELEGRAM_CHAT_ID。

使用（项目根目录）：
  .venv/bin/python watch_stock/send_analysis_to_telegram.py [yyyy-mm-dd 或 report 文件路径]
  不传参数则发送当日 watch_stock/report/yyyy-mm-dd_analysis.md
"""
import os
import sys
from datetime import date

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(REPO_ROOT, "watch_stock", "report")
DEFAULT_SUFFIX = "_analysis.md"


def resolve_report_path(arg: str | None) -> str | None:
    """
    解析出要发送的 report 文件绝对路径。
    - 无参数：当日 report/yyyy-mm-dd_analysis.md
    - yyyy-mm-dd：该日 report/yyyy-mm-dd_analysis.md
    - 文件路径：直接使用（支持相对路径，相对项目根）
    """
    if not arg or not arg.strip():
        date_str = date.today().strftime("%Y-%m-%d")
        path = os.path.join(REPORT_DIR, f"{date_str}{DEFAULT_SUFFIX}")
        return os.path.abspath(path)

    arg = arg.strip()
    # 形如 yyyy-mm-dd
    if len(arg) == 10 and arg[4] == "-" and arg[7] == "-":
        path = os.path.join(REPORT_DIR, f"{arg}{DEFAULT_SUFFIX}")
        return os.path.abspath(path)

    # 文件路径（相对项目根或绝对）
    if not os.path.isabs(arg):
        path = os.path.join(REPO_ROOT, arg)
    else:
        path = arg
    return os.path.abspath(path)


def main():
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("请设置环境变量 TELEGRAM_TOKEN、TELEGRAM_CHAT_ID")
        sys.exit(1)

    arg = sys.argv[1] if len(sys.argv) >= 2 else None
    file_path = resolve_report_path(arg)
    if not file_path or not os.path.isfile(file_path):
        print(f"报告文件不存在: {file_path or arg}")
        sys.exit(1)

    sys.path.insert(0, REPO_ROOT)
    from backtesting.notifier import TelegramNotifier

    notifier = TelegramNotifier(token=token, chat_id=chat_id)
    caption = os.path.basename(file_path)
    if notifier.send_document(file_path, caption=caption):
        print(f"已发送文档: {file_path}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
