# Reinforcement Learning-Based Adaptive AI Agents for Constructive Tactical Scenario Simulation: A 2D Pursuit-Evasion Approach

A 2D pursuit-evasion simulation project exploring reinforcement learning, agent behavior, and strategy comparison using Proximal Policy Optimization (PPO).

## 1. Project Overview

This project investigates how reinforcement learning can support adaptive decision-making in a 2D pursuit-evasion environment.

The simulation consists of:

- **Blue Agent — Pursuer:** Attempts to catch the Red agent.
- **Red Agent — Evader:** Attempts to evade the pursuer.
- **Obstacles:** Introduce navigation challenges.
- **Safe Zone:** Represents the evader's target area.
- **PPO Agent:** Learns movement decisions through interaction with the environment and reward feedback.

Three strategies—Random, Scripted, and PPO—are implemented for comparative evaluation.

## 2. Problem Statement

Traditional rule-based agents follow predefined movement strategies and may struggle to adapt to changing environmental conditions. This project explores reinforcement learning as an approach to improving evader decision-making in a simulated pursuit-evasion scenario.

The objective is to develop a 2D simulation and compare different evasion strategies using measurable performance indicators.

## 3. Objectives

- Develop a 2D pursuit-evasion simulation environment.
- Represent Blue as the pursuer and Red as the evader.
- Incorporate obstacles and a safe-zone objective.
- Implement Random and Scripted evasion strategies.
- Train a Red evader using Proximal Policy Optimization.
- Evaluate agent performance over multiple episodes.
- Compare strategies using escape rate, capture rate, survival time, reward, and trajectory data.
- Visualize the simulation through a browser-based interface.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Gymnasium | Reinforcement learning environment |
| Stable-Baselines3 | PPO implementation |
| PyTorch | Deep learning framework |
| FastAPI | Backend API |
| HTML | Frontend structure |
| CSS | Frontend styling |
| JavaScript | Simulation interaction and rendering |
| Canvas | 2D visualization |
| NumPy | Numerical computation |
| Git and GitHub | Version control and source hosting |

## 5. System Architecture

The system is organized into the following components:

1. **Frontend:** Displays the simulation and provides user controls.
2. **FastAPI Backend:** Provides endpoints for simulation interaction.
3. **Simulation Environment:** Manages agent positions, movement, observations, rewards, and episode termination.
4. **Agent Strategies:** Implements Random, Scripted, and PPO-based decision-making.
5. **PPO Training:** Trains and saves the reinforcement learning model.
6. **Evaluation Module:** Measures strategy performance.
7. **Results Storage:** Stores experimental results in CSV and JSON formats.

### Architecture Diagram

```text
                Browser Frontend
                       |
                       v
                 FastAPI Backend
                       |
                       v
             Pursuit-Evasion Environment
                       |
             +---------+---------+
             |                   |
             v                   v
        Blue Pursuer         Red Evader
                                  |
                    +-------------+-------------+
                    |             |             |
                    v             v             v
                  Random       Scripted        PPO
                                                |
                                                v
                                         Trained Model
                                                |
                                                v
                                      Evaluation Module
                                                |
                                                v
                                      CSV / JSON Results
```

## 6. Methodology

The project follows these stages:

1. Develop the 2D simulation environment.
2. Define the observation space, action space, reward function, and termination conditions.
3. Implement Random and Scripted evasion strategies.
4. Train the PPO-based Red evader.
5. Evaluate the strategies under the selected environment conditions.
6. Record performance metrics and trajectory information.
7. Compare the measured results and analyze differences in agent behavior.

### Random Strategy

The evader selects movement actions randomly from the available action space.

### Scripted Strategy

The evader follows predefined decision rules to select its movements.

### PPO Strategy

The evader uses Proximal Policy Optimization, a reinforcement learning algorithm, to learn movement decisions through environmental interaction and reward feedback.

## 7. Reinforcement Learning Approach

PPO is used to train the Red evader.

The training process involves:

- Observing the current environment state.
- Selecting a movement action.
- Receiving a reward based on the environment's reward function.
- Updating the policy using PPO.
- Repeating the process over training episodes.
- Saving the trained model for evaluation.

The intended objective is to investigate whether a learned policy can produce more adaptive behavior than the Random and Scripted strategies.

## 8. Performance Metrics

The following metrics are used to evaluate the strategies.

| Metric | Description |
|---|---|
| Escape Rate | Percentage of episodes in which the evader successfully escapes |
| Capture Rate | Percentage of episodes in which the evader is captured |
| Survival Time | Number of simulation steps completed before termination |
| Average Reward | Mean episode reward |
| Final Distance | Distance between Blue and Red at the end of an episode |
| Trajectory Analysis | Recorded movement paths of the agents |
| Safe-Zone Success Rate | Percentage of episodes in which Red reaches the safe zone |

The metrics should be interpreted according to the environment's implemented termination rules and evaluation procedure.

## 9. Project Structure

```text
Pursuit-Evasion-RL-Simulation/
├── backend/
│   ├── agents/
│   │   ├── ppo_evader.py
│   │   ├── random_evader.py
│   │   └── scripted_evader.py
│   ├── environment/
│   │   ├── pursuit_evasion_env.py
│   │   └── rl_env.py
│   ├── main.py
│   ├── train_ppo.py
│   ├── evaluate.py
│   ├── compare_strategies.py
│   └── export_ppo_csv.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/
│   ├── ppo_red_evader.zip
│   ├── ppo_red_evader_best_57.zip
│   └── ppo_red_evader_best_67.zip
├── results/
│   ├── final_results_summary.json
│   ├── ppo_evaluation.csv
│   └── strategy_comparison.json
├── tests/
│   └── test_environment.py
├── .gitignore
└── README.md
```

## 10. Installation and Setup

### Prerequisites

- Python 3.10
- Git
- A modern web browser

### Step 1: Clone the Repository

```bash
git clone https://github.com/Varshini0205/Pursuit-Evasion-RL-Simulation.git
cd Pursuit-Evasion-RL-Simulation
```

### Step 2: Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install fastapi uvicorn gymnasium==1.2.2 numpy stable-baselines3 pytest
```

PyTorch is installed as a dependency of Stable-Baselines3 when a compatible package is available for the selected Python environment.

## 11. Running the Application

### Start the Backend

From the project root, run:

```bash
python -m uvicorn backend.main:app --reload
```

Backend address:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs

### Start the Frontend

Open a second terminal from the project root and run:

```bash
python -m http.server 5500 --directory frontend
```

Open the frontend in a browser:

http://127.0.0.1:5500

Keep both terminal windows open while using the application.

## 12. Training the PPO Model

Run the training script from the project root:

```bash
python backend/train_ppo.py
```

The script trains the PPO model and saves it according to the output path configured in the training code.

Check the training script to confirm the model filename and save location before running it.

## 13. Evaluation and Results

Run the evaluation script:

```bash
python backend/evaluate.py
```

Compare the strategies:

```bash
python backend/compare_strategies.py
```

Export PPO evaluation data:

```bash
python backend/export_ppo_csv.py
```

The `results/` directory contains the project's stored evaluation summaries and experimental data.

Results should be regenerated whenever the environment, reward function, trained model, or evaluation procedure changes. Only verified experimental results should be reported in academic submissions.

## 14. Testing

Run the available environment tests:

```bash
python -m pytest tests/ -v
```

The tests help check the implemented environment behavior and identify issues during development.

## 15. Research Significance

This project provides an experimental platform for studying reinforcement learning in a simplified 2D pursuit-evasion environment.

By comparing Random, Scripted, and PPO strategies, the project investigates differences in agent behavior and performance under selected simulation conditions.

The results can help illustrate the role of reinforcement learning in sequential decision-making and provide a foundation for further experiments involving navigation, reward design, and adaptive behavior.

## 16. Potential Applications

Potential educational and research applications include:

- Reinforcement learning demonstrations.
- Autonomous-agent simulation studies.
- Path-planning and obstacle-avoidance research.
- Decision-making algorithm comparisons.
- Interactive AI education.
- Simulation-based performance evaluation.

These are potential application areas rather than claims of validated real-world deployment.

## 17. Future Improvements

Potential improvements include:

- More robust obstacle-aware navigation.
- Smoother and more realistic agent movement.
- Improved reward-function design.
- Additional PPO training and hyperparameter experiments.
- Larger and reproducible evaluation runs.
- Improved trajectory visualization.
- Expanded automated testing.
- Better integration between the frontend, backend, and trained models.

## 18. Disclaimer

This project is developed for academic and educational research into reinforcement learning and 2D agent simulation.

---

**Keywords:** Reinforcement Learning, Proximal Policy Optimization, PPO, Pursuit-Evasion, Gymnasium, Stable-Baselines3, FastAPI, 2D Simulation, Autonomous Agents, Obstacle Avoidance.
