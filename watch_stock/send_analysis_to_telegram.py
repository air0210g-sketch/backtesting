"""
将当日分析报告以 md 文档形式直接发送到 Telegram。

依赖：与 backtesting.notifier 相同（python-telegram-bot）。
配置：环境变量 TELEGRAM_TOKEN、TELEGRAM_CHAT_ID；若未设置则从项目根 .env 文件读取（KEY=VALUE 格式）。

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

TELEGRAM_ENV_KEYS = ("TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID")


def _load_env_from_dotenv(keys: tuple[str, ...]) -> None:
    """若环境变量未设置，则从项目根目录 .env 文件加载（仅补全未设置的 key）。"""
    if all(os.environ.get(k) for k in keys):
        return
    env_file = os.path.join(REPO_ROOT, ".env")
    if not os.path.isfile(env_file):
        return
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                if k in keys and not os.environ.get(k) and v.strip():
                    os.environ[k] = v.strip()
    except OSError:
        pass


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


def send_report(file_path: str) -> bool:
    """
    将指定 report 文件以文档形式发送到 Telegram。
    file_path 可为绝对路径或相对项目根的路径。依赖环境变量 TELEGRAM_TOKEN、TELEGRAM_CHAT_ID（或项目根 .env）。
    若未配置或文件不存在则静默返回 False；发送成功返回 True。供定时任务等调用。
    """
    _load_env_from_dotenv(TELEGRAM_ENV_KEYS)
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    path = file_path if os.path.isabs(file_path) else os.path.join(REPO_ROOT, file_path)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return False
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from backtesting.notifier import TelegramNotifier
    notifier = TelegramNotifier(token=token, chat_id=chat_id)
    return notifier.send_document(path, caption=os.path.basename(path))


def main():
    _load_env_from_dotenv(TELEGRAM_ENV_KEYS)
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

    if REPO_ROOT not in sys.path:
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
