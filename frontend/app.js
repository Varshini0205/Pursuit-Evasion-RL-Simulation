const API = "http://127.0.0.1:8000";

const canvas = document.getElementById("simulationCanvas");
const ctx = canvas.getContext("2d");

const strategySelect = document.getElementById("strategySelect");
const strategyElement = document.getElementById("strategy");
const stepElement = document.getElementById("step");
const statusElement = document.getElementById("status");
const blueDistanceElement = document.getElementById("blueDistance");
const safeDistanceElement = document.getElementById("safeDistance");

let simulationRunning = false;
let stepping = false;
let lastStepTime = 0;
let currentStrategy = "RANDOM";
const STEP_INTERVAL = 1000 / 30;

async function request(path, method = "GET") {
    const response = await fetch(API + path, { method });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }

    return response.json();
}

function updateMetrics(state) {
    if (state.strategy) {
        currentStrategy = state.strategy;
    }

    strategyElement.textContent = currentStrategy;
    strategySelect.value = currentStrategy;

    stepElement.textContent = state.step;
    statusElement.textContent = state.status;
    blueDistanceElement.textContent = state.distance_to_blue.toFixed(1);
    safeDistanceElement.textContent = state.distance_to_safe_zone.toFixed(1);
}

function draw(state) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = "#14532d";
    ctx.beginPath();
    ctx.arc(
        state.safe_zone.x,
        state.safe_zone.y,
        state.safe_zone.radius,
        0,
        Math.PI * 2
    );
    ctx.fill();

    ctx.fillStyle = "#4b5563";

    for (const obstacle of state.obstacles) {
        ctx.fillRect(
            obstacle.x,
            obstacle.y,
            obstacle.width,
            obstacle.height
        );
    }

    ctx.fillStyle = "#2563eb";
    ctx.beginPath();
    ctx.arc(state.blue.x, state.blue.y, 14, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#dc2626";
    ctx.beginPath();
    ctx.arc(state.red.x, state.red.y, 14, 0, Math.PI * 2);
    ctx.fill();
}

async function refresh() {
    const state = await request("/simulation/state");
    updateMetrics(state);
    draw(state);
}

async function simulationLoop(timestamp) {
    if (
        simulationRunning &&
        !stepping &&
        timestamp - lastStepTime >= STEP_INTERVAL
    ) {
        stepping = true;
        lastStepTime = timestamp;

        try {
            const state = await request("/simulation/step", "POST");

            updateMetrics(state);
            draw(state);

            if (state.done) {
                simulationRunning = false;
            }
        } catch (error) {
            console.error(error);
            simulationRunning = false;
        } finally {
            stepping = false;
        }
    }

    requestAnimationFrame(simulationLoop);
}

strategySelect.onchange = async () => {
    simulationRunning = false;

    const strategy = strategySelect.value;

    try {
        const result = await request(
            "/simulation/strategy?strategy=" + encodeURIComponent(strategy),
            "POST"
        );

        if (result.status === "ok") {
            currentStrategy = result.strategy;
            strategyElement.textContent = currentStrategy;
            strategySelect.value = currentStrategy;
        }
    } catch (error) {
        console.error(error);
    }
};

document.getElementById("startBtn").onclick = async () => {
    try {
        const state = await request("/simulation/start", "POST");

        simulationRunning = true;
        updateMetrics(state);
        draw(state);
    } catch (error) {
        console.error(error);
    }
};

document.getElementById("pauseBtn").onclick = async () => {
    const state = await request("/simulation/pause", "POST");

    simulationRunning = false;
    updateMetrics(state);
    draw(state);
};

document.getElementById("continueBtn").onclick = async () => {
    const state = await request("/simulation/resume", "POST");

    simulationRunning = true;
    updateMetrics(state);
    draw(state);
};


refresh();
requestAnimationFrame(simulationLoop);

