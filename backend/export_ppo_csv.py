import csv
from pathlib import Path
from backend.evaluate import evaluate_ppo

results = evaluate_ppo(episodes=100, seed_start=1000)

output = Path("results/ppo_evaluation.csv")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results.keys())
    writer.writeheader()
    writer.writerow(results)

print("LIVE PPO CSV EXPORT: PASSED")
print(f"Saved: {output}")
