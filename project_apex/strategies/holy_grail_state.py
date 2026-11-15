"""
Project APEX — Holy Grail State Manager

Provides persistent state storage with configurable Time-To-Live (TTL) expiration.
If the bot restarts or reconnects within TTL (default: 30 seconds), state is restored.
If downtime exceeds TTL or failed reconnects reach threshold, state is automatically
discarded to protect capital, restarting completely fresh at the base stake.
"""
import json
import os
import time
from typing import Any, Dict, Optional
from loguru import logger


class HolyGrailStateManager:
    """Manages Holy Grail recovery and active patterns state with TTL auto-reset."""

    def __init__(self, state_file: str = "datasets/holy_grail_state.json", ttl_seconds: float = 30.0):
        self.state_file = state_file
        self.ttl_seconds = ttl_seconds
        self._last_save_time: float = 0.0

    def save(self, strategy_state: Dict[str, Any], risk_state: Dict[str, Any]) -> None:
        """Persist current state with timestamp."""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            payload = {
                "timestamp": time.time(),
                "strategy": strategy_state,
                "risk": risk_state,
            }
            tmp_file = f"{self.state_file}.tmp"
            with open(tmp_file, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_file, self.state_file)
            self._last_save_time = payload["timestamp"]
        except Exception as e:
            logger.error(f"[HolyGrailStateManager] Failed to save state: {e}")

    def touch(self) -> None:
        """Heartbeat: updates saved timestamp during active operation."""
        now = time.time()
        # Only touch every 5 seconds to reduce unnecessary disk writes
        if now - self._last_save_time < 5.0 or not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
            data["timestamp"] = now
            tmp_file = f"{self.state_file}.tmp"
            with open(tmp_file, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_file, self.state_file)
            self._last_save_time = now
        except Exception:
            pass

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Load persisted state if file exists and age <= ttl_seconds.
        Returns None and clears file if state has expired beyond TTL.
        """
        if not os.path.exists(self.state_file):
            return None

        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)

            saved_time = float(data.get("timestamp", 0.0))
            age = time.time() - saved_time

            if age <= self.ttl_seconds:
                logger.success(
                    f"[HolyGrailStateManager] Restoring state (age={age:.1f}s <= {self.ttl_seconds}s TTL). "
                    f"Resuming recovery cycle."
                )
                return data
            else:
                logger.warning(
                    f"[HolyGrailStateManager] Stored state expired (age={age:.1f}s > {self.ttl_seconds}s TTL). "
                    f"Downtime exceeded threshold — discarding stale recovery state and starting FRESH."
                )
                self.clear()
                return None
        except Exception as e:
            logger.error(f"[HolyGrailStateManager] Failed to load state: {e}. Starting fresh.")
            self.clear()
            return None

    def clear(self) -> None:
        """Remove state file."""
        try:
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
        except Exception as e:
            logger.warning(f"[HolyGrailStateManager] Failed to remove state file: {e}")
