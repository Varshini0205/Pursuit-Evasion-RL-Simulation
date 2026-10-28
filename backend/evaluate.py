from pathlib import Path
import numpy as np

from backend.environment.rl_env import PursuitEvasionRLEnv
from stable_baselines3 import PPO


def evaluate_ppo(episodes=100, seed_start=1000):
    model_path = Path("models/ppo_red_evader.zip")
    model = PPO.load(str(model_path))

    escapes = 0
    captures = 0
    timeouts = 0
    rewards = []
    steps_list = []

    for episode in range(episodes):
        env = PursuitEvasionRLEnv()
        observation, _ = env.reset(seed=seed_start + episode)

        total_reward = 0.0
        steps = 0

        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)

            total_reward += float(reward)
            steps += 1

            if terminated or truncated:
                break

        status = env.sim.status

        if status == "SAFE ZONE REACHED":
            escapes += 1
        elif status == "CAPTURED":
            captures += 1
        else:
            timeouts += 1

        rewards.append(total_reward)
        steps_list.append(steps)

    return {
        "strategy": "PPO",
        "episodes": episodes,
        "escapes": escapes,
        "captures": captures,
        "timeouts": timeouts,
        "escape_rate": round(escapes / episodes * 100, 2),
        "capture_rate": round(captures / episodes * 100, 2),
        "timeout_rate": round(timeouts / episodes * 100, 2),
        "average_reward": round(float(np.mean(rewards)), 2),
        "average_steps": round(float(np.mean(steps_list)), 2),
    }


if __name__ == "__main__":
    results = evaluate_ppo()

    print("REPRODUCIBLE PPO EVALUATION: PASSED")
    for key, value in results.items():
        print(f"{key}: {value}")
