from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from backend.environment.pursuit_evasion_env import PursuitEvasionEnvironment


class PursuitEvasionRLEnv(gym.Env):
    """Gymnasium wrapper for training the Red PPO evader."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.sim = PursuitEvasionEnvironment(seed=seed)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(14,),
            dtype=np.float32,
        )

    def _get_clearances(self) -> tuple[float, float, float, float]:
        red = self.sim.red.position
        radius = self.sim.AGENT_RADIUS if hasattr(self.sim, "AGENT_RADIUS") else 14.0

        left = red.x - radius
        right = 800.0 - red.x - radius
        top = red.y - radius
        bottom = 600.0 - red.y - radius

        for obstacle in self.sim.obstacles:
            left_edge = obstacle.x - radius
            right_edge = obstacle.x + obstacle.width + radius
            top_edge = obstacle.y - radius
            bottom_edge = obstacle.y + obstacle.height + radius

            if top_edge <= red.y <= bottom_edge:
                if right_edge <= red.x:
                    left = min(left, red.x - right_edge)
                if left_edge >= red.x:
                    right = min(right, left_edge - red.x)

            if left_edge <= red.x <= right_edge:
                if bottom_edge <= red.y:
                    top = min(top, red.y - bottom_edge)
                if top_edge >= red.y:
                    bottom = min(bottom, top_edge - red.y)

        return (
            max(0.0, left),
            max(0.0, right),
            max(0.0, top),
            max(0.0, bottom),
        )

    def _get_observation(self) -> np.ndarray:
        red = self.sim.red
        blue = self.sim.blue
        safe = self.sim.safe_zone_center

        left, right, top, bottom = self._get_clearances()

        values = np.array(
            [
                red.position.x / 800.0,
                red.position.y / 600.0,
                blue.position.x / 800.0,
                blue.position.y / 600.0,
                safe.x / 800.0,
                safe.y / 600.0,
                (blue.position.x - red.position.x) / 800.0,
                (blue.position.y - red.position.y) / 600.0,
                red.velocity.x / 150.0,
                red.velocity.y / 150.0,
                left / 800.0,
                right / 800.0,
                top / 600.0,
                bottom / 600.0,
            ],
            dtype=np.float32,
        )

        return np.clip(values, -1.0, 1.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.sim.rng = np.random.default_rng(seed)

        self.sim.reset()

        return self._get_observation(), {}

    def step(self, action):
        direction = np.asarray(action, dtype=np.float32)
        length = np.linalg.norm(direction)

        if length > 1.0:
            direction = direction / length

        from backend.environment.pursuit_evasion_env import Vector2

        movement = Vector2(
            float(direction[0]),
            float(direction[1]),
        )

        previous_blue_distance = (
            self.sim.red.position - self.sim.blue.position
        ).length()

        previous_safe_distance = (
            self.sim.red.position - self.sim.safe_zone_center
        ).length()

        self.sim.step(movement)

        current_blue_distance = (
            self.sim.red.position - self.sim.blue.position
        ).length()

        current_safe_distance = (
            self.sim.red.position - self.sim.safe_zone_center
        ).length()

        reward = 0.0

        reward += (
            current_blue_distance - previous_blue_distance
        ) * 0.02

        reward += (
            previous_safe_distance - current_safe_distance
        ) * 0.01

        reward -= 0.01

        terminated = self.sim.done

        truncated = (
            self.sim.step_count >= 1800
            and not terminated
        )

        if self.sim.status == "SAFE ZONE REACHED":
            reward += 100.0

        if self.sim.status == "CAPTURED":
            reward -= 100.0

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            {},
        )
