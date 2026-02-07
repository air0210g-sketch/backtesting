import requests
import os


class TelegramNotifier:
    def __init__(self, token=None, chat_id=None):
        self.token = token
        self.chat_id = chat_id
        self.base_url = (
            f"https://api.telegram.org/bot{self.token}" if self.token else None
        )

    def send_message(self, text):
        """
        Send a text message to the configured chat_id.
        """
        if not self.token or not self.chat_id:
            print(
                f"[TelegramNotifier] No token/chat_id. Skipping message: {text[:50]}..."
            )
            return False

        url = f"{self.base_url}/sendMessage"
        data = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}
        try:
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"[TelegramNotifier] Failed to send message: {e}")
            return False

    def send_image(self, image_path, caption=None):
        """
        Send an image/photo to the configured chat_id.
        """
        if not self.token or not self.chat_id:
            print(f"[TelegramNotifier] No token/chat_id. Skipping image: {image_path}")
            return False

        if not os.path.exists(image_path):
            print(f"[TelegramNotifier] Image not found: {image_path}")
            return False

        url = f"{self.base_url}/sendPhoto"
        data = {"chat_id": self.chat_id}
        if caption:
            data["caption"] = caption

        try:
            with open(image_path, "rb") as img_file:
                files = {"photo": img_file}
                response = requests.post(url, data=data, files=files, timeout=20)
                response.raise_for_status()
            return True
        except Exception as e:
            print(f"[TelegramNotifier] Failed to send image: {e}")
            return False
