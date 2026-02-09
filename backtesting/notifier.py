"""
基于 python-telegram-bot 的 Telegram 通知封装。
对外保持同步接口：send_message / send_image / send_document。
"""
import asyncio
import os

from telegram import Bot
from telegram.error import Forbidden, InvalidToken


class TelegramNotifier:
    def __init__(self, token=None, chat_id=None):
        self.token = token
        self.chat_id = str(chat_id) if chat_id is not None else None
        self._bot = Bot(token=self.token) if self.token else None
        self._loop = None

    def _run(self, coro):
        """在同步上下文中执行异步调用。复用同一 event loop，避免多次 asyncio.run() 导致 Event loop is closed。"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = None
        except RuntimeError:
            loop = None
        if loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        self._loop = loop
        return loop.run_until_complete(coro)

    def send_message(self, text, parse_mode="HTML"):
        """
        发送文本到配置的 chat_id。

        :param text: 消息内容
        :param parse_mode: Telegram 解析模式，可选 "HTML"、"Markdown" 或 "MarkdownV2"；None 表示纯文本
        """
        if not self._bot or not self.chat_id:
            print(
                f"[TelegramNotifier] No token/chat_id. Skipping message: {text[:50]}..."
            )
            return False

        async def _send():
            kwargs = {"chat_id": self.chat_id, "text": text}
            if parse_mode:
                kwargs["parse_mode"] = parse_mode
            await self._bot.send_message(**kwargs)

        try:
            self._run(_send())
            return True
        except InvalidToken:
            print(
                "[TelegramNotifier] Unauthorized: Token 无效或已失效。"
                " 请检查 TELEGRAM_TOKEN：无空格、从 @BotFather 获取、未泄露后未重置。"
            )
            return False
        except Forbidden as e:
            print(f"[TelegramNotifier] Forbidden: {e}（例如 bot 未被用户/群组允许或已封禁）")
            return False
        except Exception as e:
            print(f"[TelegramNotifier] Failed to send message: {e}")
            return False

    def send_image(self, image_path, caption=None):
        """发送图片到配置的 chat_id。"""
        if not self._bot or not self.chat_id:
            print(f"[TelegramNotifier] No token/chat_id. Skipping image: {image_path}")
            return False

        if not os.path.exists(image_path):
            print(f"[TelegramNotifier] Image not found: {image_path}")
            return False

        async def _send():
            with open(image_path, "rb") as f:
                await self._bot.send_photo(
                    chat_id=self.chat_id,
                    photo=f,
                    caption=caption,
                )

        try:
            self._run(_send())
            return True
        except Exception as e:
            print(f"[TelegramNotifier] Failed to send image: {e}")
            return False

    def send_document(self, file_path, caption=None):
        """发送文档（如 .md 文件）到配置的 chat_id。"""
        if not self._bot or not self.chat_id:
            print(f"[TelegramNotifier] No token/chat_id. Skipping document: {file_path}")
            return False

        if not os.path.exists(file_path):
            print(f"[TelegramNotifier] File not found: {file_path}")
            return False

        async def _send():
            with open(file_path, "rb") as f:
                await self._bot.send_document(
                    chat_id=self.chat_id,
                    document=f,
                    filename=os.path.basename(file_path),
                    caption=caption,
                )

        try:
            self._run(_send())
            return True
        except Exception as e:
            print(f"[TelegramNotifier] Failed to send document: {e}")
            return False
