# utils/logger.py
import requests
import os
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

if DISCORD_WEBHOOK_URL is None:
    print("Warning: DISCORD_WEBHOOK_URL not found. Discord logging will be disabled.")

def log_to_discord(message):
    if not DISCORD_WEBHOOK_URL or DISCORD_WEBHOOK_URL == "https://discord.com/api/webhooks/1368920640429621248/g-cY5u4jgilhK-IzJH-2sqtndQi1onvUg7kew3KQ0la7RwZmFkFIc4sQBXFhqCiIwq1g": # Check if it's the placeholder
        print(f"[INFO] Discord webhook URL not configured or is placeholder. Skipping log: {message}")
        return

    try:
        payload = {
            "content": message
        }
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors (4XX or 5XX)
        # print(f"[📡] Logged to Discord: {message}") # Less verbose if successful
    except requests.exceptions.RequestException as e:
        print(f"[❌] Discord log failed: {e}")
    except Exception as e: # Catch any other unexpected errors
        print(f"[❌] An unexpected error occurred while logging to Discord: {e}")