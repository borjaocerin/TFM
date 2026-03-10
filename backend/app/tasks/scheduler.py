from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler


def create_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="Europe/Madrid")
    return scheduler
