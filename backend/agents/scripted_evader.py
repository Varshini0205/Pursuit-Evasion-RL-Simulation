from __future__ import annotations

from backend.environment.pursuit_evasion_env import Vector2


class ScriptedEvader:
    """Moves directly toward the Safe Zone."""

    def act(self, red_position: Vector2, safe_zone_position: Vector2) -> Vector2:
        return (safe_zone_position - red_position).normalized()
