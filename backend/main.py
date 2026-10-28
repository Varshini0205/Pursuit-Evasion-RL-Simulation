from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import random

from backend.environment.pursuit_evasion_env import PursuitEvasionEnvironment, Vector2
from backend.agents.ppo_evader import PPOEvader

app = FastAPI(title="Pursuit Evasion AI")

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount(
    "/frontend",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="frontend",
)

env = PursuitEvasionEnvironment()
current_strategy = "RANDOM"
simulation_running = False
ppo_agent = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "pursuit-evasion-ai"}


@app.get("/")
def frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/simulation/strategy")
def simulation_strategy(strategy: str):
    global current_strategy, ppo_agent

    strategy = strategy.upper()

    if strategy not in ["PPO", "RANDOM", "SCRIPTED"]:
        return {
            "status": "error",
            "message": "Invalid strategy. Use one of: ['PPO', 'RANDOM', 'SCRIPTED']",
        }

    current_strategy = strategy

    if strategy == "PPO":
        if ppo_agent is None:
            ppo_agent = PPOEvader(
                str(BASE_DIR / "models" / "ppo_red_evader.zip")
            )
            ppo_agent.load()

    return {
        "status": "ok",
        "strategy": current_strategy,
    }


@app.post("/simulation/reset")
def simulation_reset():
    global simulation_running

    env.reset()
    simulation_running = False

    state = env.get_state()
    state["strategy"] = current_strategy
    return state


@app.post("/simulation/start")
def simulation_start():
    global simulation_running

    env.reset()
    simulation_running = True

    state = env.get_state()
    state["strategy"] = current_strategy
    return state


@app.post("/simulation/pause")
def simulation_pause():
    global simulation_running

    simulation_running = False

    state = env.get_state()
    state["status"] = "PAUSED"
    state["strategy"] = current_strategy

    return state


@app.post("/simulation/resume")
def simulation_resume():
    global simulation_running

    simulation_running = True

    state = env.get_state()
    state["strategy"] = current_strategy
    return state


@app.post("/simulation/step")
def simulation_step():
    if not simulation_running:
        state = env.get_state()
        state["strategy"] = current_strategy
        return state

    if current_strategy == "RANDOM":
        random_direction = random.choice(
            [(1, 0), (-1, 0), (0, 1), (0, -1)]
        )

        desired_direction = Vector2(
            float(random_direction[0]),
            float(random_direction[1]),
        )

    elif current_strategy == "SCRIPTED":
        desired_direction = (
            env.safe_zone_center - env.red.position
        ).normalized()

    else:
        from backend.environment.rl_env import PursuitEvasionRLEnv

        rl_env = PursuitEvasionRLEnv()
        rl_env.sim = env

        observation = rl_env._get_observation()
        action = ppo_agent.act(observation)

        desired_direction = Vector2(
            float(action[0]),
            float(action[1]),
        )

    red_direction = env._calculate_red_direction(desired_direction)
    env.step(red_direction)

    state = env.get_state()
    state["strategy"] = current_strategy
    return state


@app.get("/simulation/state")
def simulation_state():
    state = env.get_state()
    state["strategy"] = current_strategy
    return state


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
