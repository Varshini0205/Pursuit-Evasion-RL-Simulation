from __future__ import annotations

import random

from backend.environment.pursuit_evasion_env import Vector2


class RandomEvader:
    """Chooses a random cardinal movement direction."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self) -> Vector2:
        directions = [
            Vector2(1.0, 0.0),
            Vector2(-1.0, 0.0),
            Vector2(0.0, 1.0),
            Vector2(0.0, -1.0),
        ]
        return self.rng.choice(directions)
