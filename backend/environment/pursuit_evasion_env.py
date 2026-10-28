from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


WIDTH = 800.0
HEIGHT = 600.0

AGENT_RADIUS = 14.0

RED_MAX_SPEED = 150.0
BLUE_MAX_SPEED = 165.0

RED_ACCELERATION = 420.0
BLUE_ACCELERATION = 500.0

TURN_RATE = 7.0

CAPTURE_DISTANCE = 28.0

SAFE_ZONE_X = 680.0
SAFE_ZONE_Y = 500.0
SAFE_ZONE_RADIUS = 45.0

DEFAULT_DT = 1.0 / 30.0
MAX_EPISODE_STEPS = 1800


@dataclass
class Vector2:
    x: float
    y: float

    def copy(self) -> "Vector2":
        return Vector2(self.x, self.y)

    def distance_to(self, other: "Vector2") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vector2":
        length = self.length()

        if length < 1e-8:
            return Vector2(0.0, 0.0)

        return Vector2(self.x / length, self.y / length)

    def __add__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector2":
        return Vector2(self.x * scalar, self.y * scalar)


@dataclass
class RectObstacle:
    x: float
    y: float
    width: float
    height: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def contains_point(self, point: Vector2, padding: float = 0.0) -> bool:
        return (
            self.left - padding <= point.x <= self.right + padding
            and self.top - padding <= point.y <= self.bottom + padding
        )

    def expanded(self, padding: float) -> "RectObstacle":
        return RectObstacle(
            self.x - padding,
            self.y - padding,
            self.width + 2.0 * padding,
            self.height + 2.0 * padding,
        )


@dataclass
class AgentState:
    position: Vector2
    velocity: Vector2
    heading: Vector2


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def segment_intersects_rect(
    start: Vector2,
    end: Vector2,
    rect: RectObstacle,
    padding: float = 0.0,
) -> bool:

    expanded = rect.expanded(padding)

    dx = end.x - start.x
    dy = end.y - start.y

    t_min = 0.0
    t_max = 1.0

    if abs(dx) < 1e-10:
        if start.x < expanded.left or start.x > expanded.right:
            return False
    else:
        tx1 = (expanded.left - start.x) / dx
        tx2 = (expanded.right - start.x) / dx

        if tx1 > tx2:
            tx1, tx2 = tx2, tx1

        t_min = max(t_min, tx1)
        t_max = min(t_max, tx2)

        if t_min > t_max:
            return False

    if abs(dy) < 1e-10:
        if start.y < expanded.top or start.y > expanded.bottom:
            return False
    else:
        ty1 = (expanded.top - start.y) / dy
        ty2 = (expanded.bottom - start.y) / dy

        if ty1 > ty2:
            ty1, ty2 = ty2, ty1

        t_min = max(t_min, ty1)
        t_max = min(t_max, ty2)

        if t_min > t_max:
            return False

    return True


def line_of_sight(
    start: Vector2,
    end: Vector2,
    obstacles: Sequence[RectObstacle],
    padding: float = AGENT_RADIUS,
) -> bool:

    return not any(
        segment_intersects_rect(start, end, obstacle, padding)
        for obstacle in obstacles
    )


class PursuitEvasionEnvironment:

    def __init__(
        self,
        width: float = WIDTH,
        height: float = HEIGHT,
        seed: Optional[int] = None,
    ) -> None:

        self.width = float(width)
        self.height = float(height)

        self.rng = np.random.default_rng(seed)

        self.dt = DEFAULT_DT
        self.step_count = 0

        self.capture_distance = CAPTURE_DISTANCE

        self.safe_zone_center = Vector2(
            SAFE_ZONE_X,
            SAFE_ZONE_Y,
        )

        self.safe_zone_radius = SAFE_ZONE_RADIUS

        self.red = AgentState(
            position=Vector2(120.0, 100.0),
            velocity=Vector2(0.0, 0.0),
            heading=Vector2(1.0, 0.0),
        )

        self.blue = AgentState(
            position=Vector2(660.0, 100.0),
            velocity=Vector2(0.0, 0.0),
            heading=Vector2(-1.0, 0.0),
        )

        self.obstacles: List[RectObstacle] = []

        self.done = False
        self.status = "RUNNING"

        self.last_red_collision = False
        self.last_blue_collision = False

        self.trajectory: List[dict] = []

        self.reset()

    def _create_default_obstacles(self) -> List[RectObstacle]:

        return [
            RectObstacle(250.0, 70.0, 70.0, 210.0),
            RectObstacle(470.0, 60.0, 75.0, 180.0),
            RectObstacle(190.0, 350.0, 90.0, 150.0),
            RectObstacle(390.0, 300.0, 80.0, 190.0),
            RectObstacle(570.0, 360.0, 80.0, 100.0),
        ]

    def _position_is_valid(
        self,
        position: Vector2,
        obstacles: Sequence[RectObstacle],
    ) -> bool:

        boundary_margin = AGENT_RADIUS + 2.0

        if position.x < boundary_margin:
            return False

        if position.x > self.width - boundary_margin:
            return False

        if position.y < boundary_margin:
            return False

        if position.y > self.height - boundary_margin:
            return False

        if any(
            obstacle.contains_point(position, AGENT_RADIUS)
            for obstacle in obstacles
        ):
            return False

        if (
            position.distance_to(self.safe_zone_center)
            <= self.safe_zone_radius + AGENT_RADIUS
        ):
            return False

        return True

    def _generate_obstacles(self) -> List[RectObstacle]:
        """Generate a substantially different valid obstacle layout for each reset."""
        obstacles: List[RectObstacle] = []

        protected_points = [
            Vector2(100.0, 100.0),   # Red start
            Vector2(680.0, 120.0),   # Blue start
            self.safe_zone_center,   # Safe Zone
        ]

        max_attempts = 1000

        for _ in range(max_attempts):
            if len(obstacles) >= 5:
                break

            width = float(self.rng.integers(60, 111))
            height = float(self.rng.integers(90, 191))

            x = float(
                self.rng.integers(
                    50,
                    int(WIDTH - width - 50) + 1
                )
            )

            y = float(
                self.rng.integers(
                    40,
                    int(HEIGHT - height - 40) + 1
                )
            )

            candidate = RectObstacle(
                x=x,
                y=y,
                width=width,
                height=height,
            )

            # Keep a clear area around initial agents and Safe Zone.
            protected = False

            for point in protected_points:
                if candidate.contains_point(
                    point,
                    AGENT_RADIUS + 35.0
                ):
                    protected = True
                    break

            if protected:
                continue

            # Keep reasonable spacing between obstacles.
            candidate_expanded = candidate.expanded(18.0)

            overlaps_existing = any(
                not (
                    candidate_expanded.right < obstacle.expanded(18.0).left
                    or candidate_expanded.left > obstacle.expanded(18.0).right
                    or candidate_expanded.bottom < obstacle.expanded(18.0).top
                    or candidate_expanded.top > obstacle.expanded(18.0).bottom
                )
                for obstacle in obstacles
            )

            if overlaps_existing:
                continue

            obstacles.append(candidate)

        if len(obstacles) < 5:
            raise RuntimeError(
                "Could not generate five valid random obstacles."
            )

        return obstacles
    def reset(
        self,
        seed: Optional[int] = None,
    ) -> dict:

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.done = False
        self.status = "RUNNING"

        self.last_red_collision = False
        self.last_blue_collision = False

        self.obstacles = self._generate_obstacles()

        red_candidates = [
            Vector2(100.0, 100.0),
            Vector2(120.0, 160.0),
            Vector2(100.0, 240.0),
        ]

        blue_candidates = [
            Vector2(680.0, 120.0),
            Vector2(690.0, 200.0),
            Vector2(700.0, 280.0),
        ]

        self.red.position = next(
            position
            for position in red_candidates
            if self._position_is_valid(
                position,
                self.obstacles,
            )
        )

        self.blue.position = next(
            position
            for position in blue_candidates
            if self._position_is_valid(
                position,
                self.obstacles,
            )
        )

        self.red.velocity = Vector2(0.0, 0.0)
        self.blue.velocity = Vector2(0.0, 0.0)

        self.red.heading = (
            self.safe_zone_center - self.red.position
        ).normalized()

        self.blue.heading = (
            self.red.position - self.blue.position
        ).normalized()

        self.trajectory = []

        self._record_trajectory(reward=0.0)

        return self.get_state()

    def _smooth_direction(
        self,
        current_heading: Vector2,
        desired_direction: Vector2,
    ) -> Vector2:

        desired = desired_direction.normalized()

        if desired.length() < 1e-8:
            return current_heading.normalized()

        blend = clamp(
            TURN_RATE * self.dt,
            0.0,
            1.0,
        )

        blended = Vector2(
            current_heading.x * (1.0 - blend)
            + desired.x * blend,
            current_heading.y * (1.0 - blend)
            + desired.y * blend,
        )

        return blended.normalized()

    def _attempt_move(
        self,
        agent: AgentState,
        desired_direction: Vector2,
        max_speed: float,
        acceleration: float,
    ) -> bool:
        if desired_direction.length() < 1e-8:
            desired_direction = agent.heading

        agent.heading = self._smooth_direction(
            agent.heading,
            desired_direction,
        )

        desired_velocity = agent.heading * max_speed

        velocity_delta = desired_velocity - agent.velocity
        delta_length = velocity_delta.length()
        max_delta = acceleration * self.dt

        if delta_length > max_delta:
            velocity_delta = (
                velocity_delta.normalized()
                * max_delta
            )

        agent.velocity = agent.velocity + velocity_delta

        proposed_position = (
            agent.position
            + agent.velocity * self.dt
        )

        proposed_position.x = clamp(
            proposed_position.x,
            AGENT_RADIUS,
            self.width - AGENT_RADIUS,
        )

        proposed_position.y = clamp(
            proposed_position.y,
            AGENT_RADIUS,
            self.height - AGENT_RADIUS,
        )

        blocked = any(
            segment_intersects_rect(
                agent.position,
                proposed_position,
                obstacle,
                AGENT_RADIUS,
            )
            for obstacle in self.obstacles
        )

        if blocked:
            agent.velocity = agent.velocity * 0.15

            escape_directions = [
                Vector2(-agent.heading.y, agent.heading.x),
                Vector2(agent.heading.y, -agent.heading.x),
            ]

            for escape_direction in escape_directions:
                escape_position = (
                    agent.position
                    + escape_direction.normalized()
                    * max_speed
                    * self.dt
                )

                escape_position.x = clamp(
                    escape_position.x,
                    AGENT_RADIUS,
                    self.width - AGENT_RADIUS,
                )
                escape_position.y = clamp(
                    escape_position.y,
                    AGENT_RADIUS,
                    self.height - AGENT_RADIUS,
                )

                escape_blocked = any(
                    segment_intersects_rect(
                        agent.position,
                        escape_position,
                        obstacle,
                        AGENT_RADIUS,
                    )
                    for obstacle in self.obstacles
                )

                if not escape_blocked:
                    agent.position = escape_position
                    agent.heading = escape_direction.normalized()
                    agent.velocity = (
                        agent.heading
                        * min(max_speed * 0.35, max_speed)
                    )
                    return True

            return False

        agent.position = proposed_position
        return True

    def _blue_candidate_directions(self) -> List[Vector2]:

        target = self.red.position
        origin = self.blue.position

        direct = (target - origin).normalized()

        if direct.length() < 1e-8:
            direct = self.blue.heading

        return [
            direct,
            Vector2(direct.y, -direct.x).normalized(),
            Vector2(-direct.y, direct.x).normalized(),
            Vector2(
                direct.x + direct.y,
                direct.y - direct.x,
            ).normalized(),
            Vector2(
                direct.x - direct.y,
                direct.y + direct.x,
            ).normalized(),
            Vector2(1.0, 0.0),
            Vector2(-1.0, 0.0),
            Vector2(0.0, 1.0),
            Vector2(0.0, -1.0),
        ]

    def _score_blue_direction(
        self,
        direction: Vector2,
    ) -> float:
        direction = direction.normalized()
        look_distance = 140.0

        probe = (
            self.blue.position
            + direction * look_distance
        )

        if (
            probe.x < AGENT_RADIUS
            or probe.x > self.width - AGENT_RADIUS
            or probe.y < AGENT_RADIUS
            or probe.y > self.height - AGENT_RADIUS
        ):
            return float("inf")

        if any(
            segment_intersects_rect(
                self.blue.position,
                probe,
                obstacle,
                AGENT_RADIUS,
            )
            for obstacle in self.obstacles
        ):
            return float("inf")

        target_distance = probe.distance_to(
            self.red.position
        )

        dot = clamp(
            direction.x * self.blue.heading.x
            + direction.y * self.blue.heading.y,
            -1.0,
            1.0,
        )

        heading_penalty = (1.0 - dot) * 30.0

        clearance_penalty = 0.0

        for obstacle in self.obstacles:
            expanded = obstacle.expanded(
                AGENT_RADIUS + 12.0
            )

            closest_x = clamp(
                probe.x,
                expanded.left,
                expanded.right,
            )

            closest_y = clamp(
                probe.y,
                expanded.top,
                expanded.bottom,
            )

            distance = math.hypot(
                probe.x - closest_x,
                probe.y - closest_y,
            )

            if distance < 45.0:
                clearance_penalty += (45.0 - distance) * 2.0

        return (
            target_distance
            + heading_penalty
            + clearance_penalty
        )

    def _calculate_blue_direction(self) -> Vector2:

        direct = (
            self.red.position
            - self.blue.position
        ).normalized()

        if line_of_sight(
            self.blue.position,
            self.red.position,
            self.obstacles,
            AGENT_RADIUS,
        ):
            return direct

        candidates = self._blue_candidate_directions()

        scored = [
            (
                self._score_blue_direction(direction),
                direction,
            )
            for direction in candidates
        ]

        scored.sort(key=lambda item: item[0])

        for score, direction in scored:
            if math.isfinite(score):
                return direction

        return self.blue.heading

    def _red_candidate_directions(
        self,
        desired_direction: Vector2,
    ) -> List[Vector2]:

        desired = desired_direction.normalized()

        if desired.length() < 1e-8:
            desired = self.red.heading

        return [
            desired,
            Vector2(
                desired.y,
                -desired.x,
            ).normalized(),
            Vector2(
                -desired.y,
                desired.x,
            ).normalized(),
            Vector2(
                desired.x + desired.y,
                desired.y - desired.x,
            ).normalized(),
            Vector2(
                desired.x - desired.y,
                desired.y + desired.x,
            ).normalized(),
            Vector2(1.0, 0.0),
            Vector2(-1.0, 0.0),
            Vector2(0.0, 1.0),
            Vector2(0.0, -1.0),
        ]

    def _calculate_red_direction(
        self,
        desired_direction: Vector2,
    ) -> Vector2:

        desired = desired_direction.normalized()

        if desired.length() < 1e-8:
            desired = self.red.heading

        look_distance = 70.0

        direct_probe = (
            self.red.position
            + desired * look_distance
        )

        if (
            direct_probe.x >= AGENT_RADIUS
            and direct_probe.x <= self.width - AGENT_RADIUS
            and direct_probe.y >= AGENT_RADIUS
            and direct_probe.y <= self.height - AGENT_RADIUS
            and line_of_sight(
                self.red.position,
                direct_probe,
                self.obstacles,
                AGENT_RADIUS,
            )
        ):
            return desired

        candidates = self._red_candidate_directions(
            desired
        )

        best_direction = self.red.heading
        best_score = float("inf")

        for direction in candidates:

            probe = (
                self.red.position
                + direction * look_distance
            )

            if (
                probe.x < AGENT_RADIUS
                or probe.x > self.width - AGENT_RADIUS
                or probe.y < AGENT_RADIUS
                or probe.y > self.height - AGENT_RADIUS
            ):
                continue

            if not line_of_sight(
                self.red.position,
                probe,
                self.obstacles,
                AGENT_RADIUS,
            ):
                continue

            safe_distance = probe.distance_to(
                self.safe_zone_center
            )

            blue_distance = probe.distance_to(
                self.blue.position
            )

            heading_similarity = (
                direction.x * self.red.heading.x
                + direction.y * self.red.heading.y
            )

            score = (
                safe_distance
                - blue_distance * 0.35
                - heading_similarity * 20.0
            )

            if score < best_score:
                best_score = score
                best_direction = direction

        return best_direction

    def move_red(
        self,
        desired_direction: Vector2,
    ) -> bool:

        if self.done:
            return False

        safe_direction = self._calculate_red_direction(
            desired_direction
        )

        moved = self._attempt_move(
            self.red,
            safe_direction,
            RED_MAX_SPEED,
            RED_ACCELERATION,
        )

        self.last_red_collision = not moved

        return moved

    def update_blue(self) -> bool:

        if self.done:
            return False

        desired_direction = (
            self._calculate_blue_direction()
        )

        moved = self._attempt_move(
            self.blue,
            desired_direction,
            BLUE_MAX_SPEED,
            BLUE_ACCELERATION,
        )

        self.last_blue_collision = not moved

        return moved

    def _check_safe_zone(self) -> bool:

        return (
            self.red.position.distance_to(
                self.safe_zone_center
            )
            <= self.safe_zone_radius
        )

    def _check_capture(self) -> bool:

        return (
            self.red.position.distance_to(
                self.blue.position
            )
            <= self.capture_distance
        )

    def step(
        self,
        red_direction: Vector2,
        reward: float = 0.0,
    ) -> dict:

        if self.done:
            return self.get_state()

        self.last_red_collision = False
        self.last_blue_collision = False

        self.move_red(red_direction)
        self.update_blue()

        self.step_count += 1

        if self._check_safe_zone():

            self.done = True
            self.status = "SAFE ZONE REACHED"

        elif self._check_capture():

            self.done = True
            self.status = "CAPTURED"

        elif self.step_count >= MAX_EPISODE_STEPS:

            self.done = True
            self.status = "TIME LIMIT"

        else:

            self.status = "RUNNING"

        self._record_trajectory(reward)

        return self.get_state()

    def _record_trajectory(
        self,
        reward: float,
    ) -> None:

        distance_to_blue = (
            self.red.position.distance_to(
                self.blue.position
            )
        )

        distance_to_safe_zone = (
            self.red.position.distance_to(
                self.safe_zone_center
            )
        )

        self.trajectory.append(
            {
                "step": self.step_count,
                "blue_x": round(
                    self.blue.position.x,
                    4,
                ),
                "blue_y": round(
                    self.blue.position.y,
                    4,
                ),
                "red_x": round(
                    self.red.position.x,
                    4,
                ),
                "red_y": round(
                    self.red.position.y,
                    4,
                ),
                "distance_to_blue": round(
                    distance_to_blue,
                    4,
                ),
                "distance_to_safe_zone": round(
                    distance_to_safe_zone,
                    4,
                ),
                "reward": float(reward),
            }
        )

    def get_state(self) -> dict:

        distance_to_blue = (
            self.red.position.distance_to(
                self.blue.position
            )
        )

        distance_to_safe_zone = (
            self.red.position.distance_to(
                self.safe_zone_center
            )
        )

        return {
            "width": self.width,
            "height": self.height,
            "step": self.step_count,
            "status": self.status,
            "done": self.done,
            "blue": {
                "x": self.blue.position.x,
                "y": self.blue.position.y,
                "vx": self.blue.velocity.x,
                "vy": self.blue.velocity.y,
            },
            "red": {
                "x": self.red.position.x,
                "y": self.red.position.y,
                "vx": self.red.velocity.x,
                "vy": self.red.velocity.y,
            },
            "safe_zone": {
                "x": self.safe_zone_center.x,
                "y": self.safe_zone_center.y,
                "radius": self.safe_zone_radius,
            },
            "obstacles": [
                {
                    "x": obstacle.x,
                    "y": obstacle.y,
                    "width": obstacle.width,
                    "height": obstacle.height,
                }
                for obstacle in self.obstacles
            ],
            "distance_to_blue": distance_to_blue,
            "distance_to_safe_zone": distance_to_safe_zone,
        }


def run_stuck_prevention_test() -> None:

    print("=" * 60)
    print("BLUE STUCK-PREVENTION TEST")
    print("=" * 60)

    env = PursuitEvasionEnvironment(seed=99)

    # Construct a controlled corner scenario.
    env.obstacles = [
        RectObstacle(
            300.0,
            180.0,
            120.0,
            240.0,
        )
    ]

    env.blue.position = Vector2(
        240.0,
        300.0,
    )

    env.red.position = Vector2(
        520.0,
        300.0,
    )

    env.blue.velocity = Vector2(0.0, 0.0)
    env.blue.heading = Vector2(1.0, 0.0)

    start_position = env.blue.position.copy()

    positions = []

    for _ in range(120):

        env.update_blue()

        positions.append(
            (
                env.blue.position.x,
                env.blue.position.y,
            )
        )

    unique_positions = {
        (
            round(x, 2),
            round(y, 2),
        )
        for x, y in positions
    }

    total_displacement = env.blue.position.distance_to(
        start_position
    )

    # Blue must have moved.
    assert total_displacement > 20.0

    # Blue must not have entered the obstacle.
    assert not env.obstacles[0].contains_point(
        env.blue.position,
        AGENT_RADIUS,
    )

    # Blue must have changed its path rather than remaining
    # permanently at one position.
    assert len(unique_positions) > 10

    print("[PASS] Blue moved around the blocking obstacle")
    print("[PASS] Blue remained outside the obstacle")
    print("[PASS] Blue did not remain permanently stuck")

    print("=" * 60)
    print("BLUE STUCK-PREVENTION TEST PASSED")
    print("=" * 60)


def run_smoke_tests() -> None:

    print("=" * 60)
    print("PHASE 1 - ENVIRONMENT FOUNDATION TEST")
    print("=" * 60)

    env = PursuitEvasionEnvironment(seed=42)

    assert env.width == 800.0
    assert env.height == 600.0
    print("[PASS] Environment created: 800 x 600")

    assert env._position_is_valid(
        env.red.position,
        env.obstacles,
    )

    assert env._position_is_valid(
        env.blue.position,
        env.obstacles,
    )

    print("[PASS] Initial agent positions are valid")

    for obstacle in env.obstacles:
        assert not obstacle.contains_point(
            env.safe_zone_center,
            env.safe_zone_radius,
        )

    print("[PASS] Safe Zone is clear")

    env.obstacles = [
        RectObstacle(
            200.0,
            200.0,
            200.0,
            100.0,
        )
    ]

    env.red.position = Vector2(
        150.0,
        250.0,
    )

    env.red.velocity = Vector2(0.0, 0.0)

    env.red.heading = Vector2(1.0, 0.0)

    original_red = env.red.position.copy()

    blocked_position = Vector2(
        190.0,
        250.0,
    )

    assert segment_intersects_rect(
        original_red,
        blocked_position,
        env.obstacles[0],
        AGENT_RADIUS,
    )

    moved = env._attempt_move(
        env.red,
        Vector2(1.0, 0.0),
        RED_MAX_SPEED,
        RED_ACCELERATION,
    )

    assert moved is True
    assert env.red.position.distance_to(
        original_red
    ) > 0.0

    print("[PASS] Red movement and obstacle collision geometry work")

    env.blue.position = Vector2(
        150.0,
        250.0,
    )

    env.blue.velocity = Vector2(0.0, 0.0)

    env.blue.heading = Vector2(1.0, 0.0)

    original_blue = env.blue.position.copy()

    blocked_position = Vector2(
        190.0,
        250.0,
    )

    assert segment_intersects_rect(
        original_blue,
        blocked_position,
        env.obstacles[0],
        AGENT_RADIUS,
    )

    moved = env._attempt_move(
        env.blue,
        Vector2(1.0, 0.0),
        BLUE_MAX_SPEED,
        BLUE_ACCELERATION,
    )

    assert moved is True
    assert env.blue.position.distance_to(
        original_blue
    ) > 0.0

    print("[PASS] Blue movement and obstacle collision geometry work")

    env.obstacles = []

    env.red.position = Vector2(
        100.0,
        100.0,
    )

    env.red.velocity = Vector2(
        0.0,
        0.0,
    )

    env.red.heading = Vector2(
        1.0,
        0.0,
    )

    old_position = env.red.position.copy()

    moved = env.move_red(
        Vector2(1.0, 0.0)
    )

    assert moved is True

    assert (
        env.red.position.distance_to(
            old_position
        )
        > 0.0
    )

    assert (
        env.red.velocity.length()
        < RED_MAX_SPEED
    )

    print("[PASS] Smooth acceleration works")

    env.obstacles = [
        RectObstacle(
            250.0,
            180.0,
            180.0,
            180.0,
        )
    ]

    env.blue.position = Vector2(
        150.0,
        270.0,
    )

    env.red.position = Vector2(
        550.0,
        270.0,
    )

    env.blue.heading = Vector2(
        1.0,
        0.0,
    )

    assert not line_of_sight(
        env.blue.position,
        env.red.position,
        env.obstacles,
        AGENT_RADIUS,
    )

    direction = env._calculate_blue_direction()

    assert direction.length() > 0.0

    print("[PASS] Blue detects blocked path")

    env.red.position = Vector2(
        300.0,
        300.0,
    )

    env.blue.position = Vector2(
        300.0 + CAPTURE_DISTANCE - 1.0,
        300.0,
    )

    assert env._check_capture()

    print("[PASS] Capture detection works")

    env.red.position = (
        env.safe_zone_center.copy()
    )

    assert env._check_safe_zone()

    print("[PASS] Safe Zone detection works")

    env.reset(seed=42)

    state = env.step(
        Vector2(1.0, 0.0)
    )

    assert isinstance(state, dict)
    assert "blue" in state
    assert "red" in state
    assert "safe_zone" in state
    assert "obstacles" in state

    print("[PASS] Simulation step works")

    env_a = PursuitEvasionEnvironment(
        seed=123
    )

    env_b = PursuitEvasionEnvironment(
        seed=123
    )

    state_a = env_a.reset(seed=123)
    state_b = env_b.reset(seed=123)

    assert state_a["red"] == state_b["red"]
    assert state_a["blue"] == state_b["blue"]
    assert state_a["obstacles"] == state_b["obstacles"]

    print("[PASS] Seed reproducibility works")

    print("=" * 60)
    print("ALL PHASE 1 TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_tests()


