from __future__ import annotations

from pathlib import Path

import numpy as np


class PPOEvader:
    """Loads and uses a trained PPO model for the Red agent."""

    def __init__(self, model_path: str = "models/ppo_red_evader.zip"):
        self.model_path = Path(model_path)
        self.model = None

    def load(self) -> None:
        from stable_baselines3 import PPO

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"PPO model not found: {self.model_path}"
            )

        self.model = PPO.load(str(self.model_path))

    def act(self, observation: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PPO model is not loaded.")

        action, _ = self.model.predict(
            observation,
            deterministic=True,
        )

        return np.asarray(action)
