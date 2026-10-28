from stable_baselines3 import PPO
from backend.environment.rl_env import PursuitEvasionRLEnv

env = PursuitEvasionRLEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=42,
)

model.learn(total_timesteps=100000)
model.save("models/ppo_red_evader")

env.close()

print("PPO TRAINING SCRIPT: PASSED")
print("Saved: models/ppo_red_evader.zip")
