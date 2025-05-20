# utils/logger.py

import requests
import os

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/your_webhook_here"

def log_to_discord(message):
    try:
        payload = {
            "content": message
        }
        requests.post(DISCORD_WEBHOOK_URL, json=payload)
        print(f"[📡] Logged to Discord: {message}")
    except Exception as e:
        print(f"[❌] Discord log failed: {e}")
