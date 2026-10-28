from backend.environment.pursuit_evasion_env import (
    PursuitEvasionEnvironment,
    Vector2,
    RectObstacle,
    AGENT_RADIUS,
    BLUE_MAX_SPEED,
    BLUE_ACCELERATION,
    CAPTURE_DISTANCE,
)


def test_environment_dimensions():
    env = PursuitEvasionEnvironment(seed=1)

    assert env.width == 800.0
    assert env.height == 600.0


def test_agents_start_outside_obstacles():
    env = PursuitEvasionEnvironment(seed=1)

    for obstacle in env.obstacles:
        assert not obstacle.contains_point(
            env.red.position,
            AGENT_RADIUS,
        )
        assert not obstacle.contains_point(
            env.blue.position,
            AGENT_RADIUS,
        )


def test_red_cannot_cross_obstacle():
    env = PursuitEvasionEnvironment(seed=1)

    env.obstacles = [
        RectObstacle(
            200.0,
            200.0,
            200.0,
            100.0,
        )
    ]

    env.red.position = Vector2(150.0, 250.0)
    env.red.velocity = Vector2(0.0, 0.0)
    env.red.heading = Vector2(1.0, 0.0)

    for _ in range(30):
        env.move_red(Vector2(1.0, 0.0))

    assert not env.obstacles[0].contains_point(
        env.red.position,
        AGENT_RADIUS,
    )


def test_blue_cannot_cross_obstacle():
    env = PursuitEvasionEnvironment(seed=1)

    env.obstacles = [
        RectObstacle(
            200.0,
            200.0,
            200.0,
            100.0,
        )
    ]

    env.blue.position = Vector2(150.0, 250.0)
    env.blue.velocity = Vector2(0.0, 0.0)
    env.blue.heading = Vector2(1.0, 0.0)

    for _ in range(30):
        env._attempt_move(
            env.blue,
            Vector2(1.0, 0.0),
            BLUE_MAX_SPEED,
            BLUE_ACCELERATION,
        )

    assert not env.obstacles[0].contains_point(
        env.blue.position,
        AGENT_RADIUS,
    )


def test_capture_detection():
    env = PursuitEvasionEnvironment(seed=1)

    env.red.position = Vector2(300.0, 300.0)
    env.blue.position = Vector2(
        300.0 + CAPTURE_DISTANCE - 1.0,
        300.0,
    )

    assert env._check_capture()


def test_safe_zone_detection():
    env = PursuitEvasionEnvironment(seed=1)

    env.red.position = env.safe_zone_center.copy()

    assert env._check_safe_zone()


def test_simulation_step():
    env = PursuitEvasionEnvironment(seed=1)

    state = env.step(
        Vector2(1.0, 0.0)
    )

    assert isinstance(state, dict)
    assert state["step"] == 1
    assert "blue" in state
    assert "red" in state
    assert "safe_zone" in state
    assert "obstacles" in state


def test_seed_reproducibility():
    env_a = PursuitEvasionEnvironment(seed=42)
    env_b = PursuitEvasionEnvironment(seed=42)

    state_a = env_a.reset(seed=42)
    state_b = env_b.reset(seed=42)

    assert state_a["red"] == state_b["red"]
    assert state_a["blue"] == state_b["blue"]
    assert state_a["obstacles"] == state_b["obstacles"]
