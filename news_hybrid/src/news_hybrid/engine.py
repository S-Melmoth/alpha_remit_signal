"""Stateful causal selection, idempotency, weekly cap and push construction."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .contracts import MOSCOW, HybridRequest, canonical_json, digest, timestamp
from .news_features import aggregate_news
from .predictor import ModelRegistry
from .text import render_push

POLICY_ID = "l5-news-q85-h10-v1"
WEEKLY_CAP = 2


class HybridEngine:
    def __init__(self, state_path: str | Path, registry: ModelRegistry | None = None):
        path = Path(state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.registry = registry or ModelRegistry()
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, timeout=30, isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA busy_timeout=30000")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS decisions(
                event_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, result_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS state(
                currency TEXT PRIMARY KEY, armed INTEGER NOT NULL,
                last_decision_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS weekly_usage(
                currency TEXT NOT NULL, iso_year INTEGER NOT NULL, iso_week INTEGER NOT NULL,
                used INTEGER NOT NULL,
                PRIMARY KEY(currency, iso_year, iso_week)
            );
        """)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> "HybridEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def process(self, request: HybridRequest) -> dict:
        news, visible = aggregate_news(request.news_events, request.currency, request.decision_at)
        model = self.registry.at(request.decision_at)
        score = model.score(request.currency, request.price_features, news) if model else None
        threshold = model.threshold(request.currency) if model else None
        fingerprint = digest(request.to_dict())
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                existing = self._db.execute(
                    "SELECT fingerprint, result_json FROM decisions WHERE event_id=?",
                    (request.event_id,),
                ).fetchone()
                if existing is not None:
                    if existing["fingerprint"] != fingerprint:
                        raise ValueError("event_id was already used with different inputs")
                    self._db.execute("COMMIT")
                    return json.loads(existing["result_json"])
                result = self._decide(request, model, score, threshold, visible)
                self._db.execute(
                    "INSERT INTO decisions VALUES (?, ?, ?)",
                    (request.event_id, fingerprint, canonical_json(result)),
                )
                self._db.execute("COMMIT")
                return result
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def _decide(self, request, model, score, threshold, visible) -> dict:
        current = timestamp(request.decision_at)
        previous = self._db.execute(
            "SELECT armed, last_decision_at FROM state WHERE currency=?", (request.currency,)
        ).fetchone()
        if previous is not None and current <= previous["last_decision_at"]:
            raise ValueError("new decisions must be chronological within a currency")
        armed = bool(previous["armed"]) if previous is not None else False
        new_episode = False
        if model is None:
            armed = False
            reason = "model_outside_validity"
        elif score < threshold:
            armed = True
            reason = "low_score_rearmed"
        elif not armed:
            reason = "no_new_high_episode"
        else:
            armed = False
            new_episode = True
            if not visible:
                reason = "new_episode_without_news"
            else:
                iso = request.decision_at.astimezone(MOSCOW).isocalendar()
                row = self._db.execute(
                    "SELECT used FROM weekly_usage WHERE currency=? AND iso_year=? AND iso_week=?",
                    (request.currency, iso.year, iso.week),
                ).fetchone()
                used = int(row["used"]) if row else 0
                if used >= WEEKLY_CAP:
                    reason = "weekly_cap"
                else:
                    used += 1
                    self._db.execute(
                        "INSERT INTO weekly_usage VALUES (?, ?, ?, ?) "
                        "ON CONFLICT(currency, iso_year, iso_week) DO UPDATE SET used=excluded.used",
                        (request.currency, iso.year, iso.week, used),
                    )
                    reason = "ready"
        self._db.execute(
            "INSERT INTO state VALUES (?, ?, ?) ON CONFLICT(currency) DO UPDATE SET "
            "armed=excluded.armed, last_decision_at=excluded.last_decision_at",
            (request.currency, int(armed), current),
        )
        ready = reason == "ready"
        notification = None
        if ready:
            title, body = render_push(
                request.currency, request.rub_per_unit, request.effective_date
            )
            notification_id = digest(
                {"policy": POLICY_ID, "event_id": request.event_id, "currency": request.currency}
            )
            notification = {
                "schema_version": "transfer-notification.v1",
                "notification_id": notification_id,
                "currency": request.currency,
                "title": title,
                "body": body,
                "decision_at": current,
                "effective_date": request.effective_date.isoformat(),
                "rub_per_unit": request.rub_per_unit,
                "news": [
                    {
                        "key": item.key,
                        "headline": item.headline,
                        "available_at": timestamp(item.available_at),
                        "source_urls": list(item.source_urls),
                    }
                    for item in visible
                ],
            }
        return {
            "schema_version": "news-hybrid-decision.v1",
            "status": "READY" if ready else "ABSTAIN",
            "reason": reason,
            "policy_id": POLICY_ID,
            "event_id": request.event_id,
            "currency": request.currency,
            "decision_at": current,
            "model_version": model.version if model else None,
            "score": score,
            "threshold": threshold,
            "score_is_calibrated_probability": False,
            "new_episode": new_episode,
            "news_gate": bool(visible),
            "news_event_keys": [item.key for item in visible],
            "notification": notification,
        }
