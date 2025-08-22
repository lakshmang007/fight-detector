"""
utils.py

Helper utilities for the MVP, including notification when a fight is detected.
Replace the notification implementation with email/SMS/Push as needed.
"""
from __future__ import annotations

import datetime as _dt


def notify_fight_detected(prob: float):
    """
    Basic notification stub. Extend this to send emails, SMS, or push notifications.
    For now, prints a message with timestamp. You could also hook in playsound, etc.
    """
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[ALERT {ts}] Fight likely detected! probability={prob:.2f}")

