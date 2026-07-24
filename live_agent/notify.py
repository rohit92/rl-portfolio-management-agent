"""
notify.py — Push messages to Telegram / WhatsApp / macOS.

Channels (each enabled only if its env vars are set, so it degrades gracefully):

  • Telegram  — free, easy. Create a bot via @BotFather, get the token, then your
                chat id (message the bot, then read it from getUpdates). Set:
                  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  • WhatsApp  — via CallMeBot (free for personal use): message their number once to
                get an API key (https://www.callmebot.com/blog/free-api-whatsapp-messages/).
                Set: WHATSAPP_PHONE (e.g. +9198xxxxxxx), CALLMEBOT_APIKEY
                (For production/business use, swap in Twilio or the Meta Cloud API.)
  • macOS     — always-on local desktop notification fallback.

    python notify.py "hello from the trading agent"   # send a test to all configured
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
logger = logging.getLogger("notify")


def _telegram(msg: str) -> Optional[bool]:
    tok, cid = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not tok or not cid:
        return None
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{tok}/sendMessage",
            params={"chat_id": cid, "text": msg, "parse_mode": "HTML",
                    "disable_web_page_preview": True}, timeout=15)
        return bool(r.ok)
    except Exception as exc:
        logger.warning("Telegram send failed: %s", exc)
        return False


def _whatsapp(msg: str) -> Optional[bool]:
    phone, key = os.getenv("WHATSAPP_PHONE"), os.getenv("CALLMEBOT_APIKEY")
    if not phone or not key:
        return None
    try:
        r = requests.get("https://api.callmebot.com/whatsapp.php",
                         params={"phone": phone, "text": msg, "apikey": key}, timeout=25)
        return bool(r.ok)
    except Exception as exc:
        logger.warning("WhatsApp send failed: %s", exc)
        return False


def _macos(msg: str) -> bool:
    try:
        first = msg.splitlines()[0][:120]
        subprocess.run(["osascript", "-e",
            f'display notification "{first}" with title "Trading Agent signal"'],
            check=False, capture_output=True, timeout=10)
        return True
    except Exception:
        return False


def status() -> Dict[str, bool]:
    """Which channels are configured."""
    return {
        "telegram": bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID")),
        "whatsapp": bool(os.getenv("WHATSAPP_PHONE") and os.getenv("CALLMEBOT_APIKEY")),
        "macos": sys.platform == "darwin",
    }


def send(msg: str, channels: Optional[List[str]] = None) -> Dict[str, Optional[bool]]:
    """Send to all configured channels (or a subset). Returns per-channel result
    (True ok, False failed, None not-configured)."""
    fns = {"telegram": _telegram, "whatsapp": _whatsapp, "macos": _macos}
    targets = channels or list(fns)
    out = {c: fns[c](msg) for c in targets if c in fns}
    logger.info("notify -> %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    text = " ".join(sys.argv[1:]) or "Test message from the trading agent ✅"
    st = status()
    print("Configured channels:", {k: v for k, v in st.items() if v} or "none "
          "(set TELEGRAM_* or WHATSAPP_* in .env)")
    print("Result:", send(text))
