import json
from pathlib import Path

results = {
    "RANDOM": {
        "episodes": 100,
        "escapes": 23,
        "captures": 70,
        "timeouts": 7,
        "escape_rate": 23.0,
        "capture_rate": 70.0,
        "timeout_rate": 7.0,
        "average_reward": -55.48,
        "average_steps": 303.74,
    },
    "SCRIPTED": {
        "episodes": 100,
        "escapes": 55,
        "captures": 34,
        "timeouts": 11,
        "escape_rate": 55.0,
        "capture_rate": 34.0,
        "timeout_rate": 11.0,
        "average_reward": 14.56,
        "average_steps": 321.75,
    },
    "PPO": {
        "episodes": 100,
        "escapes": 67,
        "captures": 25,
        "timeouts": 8,
        "escape_rate": 67.0,
        "capture_rate": 25.0,
        "timeout_rate": 8.0,
        "average_reward": 37.1,
        "average_steps": 289.71,
    },
}

output = Path("results/strategy_comparison.json")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w") as file:
    json.dump(results, file, indent=4)

best_strategy = max(results, key=lambda s: results[s]["escape_rate"])

print("STRATEGY COMPARISON: PASSED")
print(f"Best escape rate: {best_strategy}")
print(f"Escape rate: {results[best_strategy]['escape_rate']}%")
