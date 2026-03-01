"""A lightweight Geometry Dash autoplayer core.

The module models the level as a 1D lane with rectangular spikes and pits.
The AI searches for a safe sequence of movement and jump actions that can
reach the finish line.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import product
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Hazard:
    """Dangerous segment on the ground."""

    start: float
    end: float

    def overlaps(self, x: float) -> bool:
        return self.start <= x <= self.end


@dataclass(frozen=True)
class Level:
    """Simplified Geometry Dash level model."""

    length: float
    hazards: Tuple[Hazard, ...]
    gravity: float = 28.0
    speed: float = 8.0
    jump_velocity: float = 10.5


@dataclass(frozen=True)
class State:
    """Discrete physics state for planning."""

    x: int
    y: int
    vy: int
    on_ground: bool


@dataclass(frozen=True)
class Action:
    """Control at one simulation step."""

    press_jump: bool

    def __str__(self) -> str:
        return "JUMP" if self.press_jump else "RUN"


@dataclass(frozen=True)
class SolveConfig:
    """Solver/physics params for one search attempt."""

    dt: float
    scale: int
    gravity: float
    speed: float
    jump_velocity: float


class GeometryDashAI:
    """Pathfinding-based AI that computes a safe action sequence."""

    def __init__(self, level: Level, dt: float = 0.05, scale: int = 100):
        self.level = level
        self.dt = dt
        self.scale = scale

    def _to_world(self, value: int) -> float:
        return value / self.scale

    def _to_grid(self, value: float) -> int:
        return round(value * self.scale)

    def _hits_hazard(self, x_world: float, y_world: float) -> bool:
        if y_world > 0.0:
            return False
        return any(h.overlaps(x_world) for h in self.level.hazards)

    def _step(self, state: State, action: Action) -> Optional[State]:
        x = self._to_world(state.x)
        y = self._to_world(state.y)
        vy = self._to_world(state.vy)

        if action.press_jump and state.on_ground:
            vy = self.level.jump_velocity

        x += self.level.speed * self.dt
        vy -= self.level.gravity * self.dt
        y += vy * self.dt

        on_ground = y <= 0.0
        if on_ground:
            y = 0.0
            vy = 0.0

        if self._hits_hazard(x, y):
            return None

        if x > self.level.length:
            x = self.level.length

        return State(
            x=self._to_grid(x),
            y=self._to_grid(y),
            vy=self._to_grid(vy),
            on_ground=on_ground,
        )

    def solve(self, max_steps: int = 2000) -> List[Action]:
        """Find a safe action sequence until the finish line.

        Uses Dijkstra (uniform edge weights) over discretized physics states.
        """

        start = State(x=0, y=0, vy=0, on_ground=True)
        target_x = self._to_grid(self.level.length)

        frontier: List[Tuple[int, int, State]] = []
        push_id = 0
        heappush(frontier, (0, push_id, start))

        dist: Dict[State, int] = {start: 0}
        parent: Dict[State, Tuple[State, Action]] = {}

        while frontier:
            cost, _, current = heappop(frontier)
            if current.x >= target_x:
                return self._reconstruct(parent, current)

            if cost >= max_steps:
                continue

            for action in (Action(False), Action(True)):
                nxt = self._step(current, action)
                if nxt is None:
                    continue

                new_cost = cost + 1
                if new_cost < dist.get(nxt, 10**9):
                    dist[nxt] = new_cost
                    parent[nxt] = (current, action)
                    push_id += 1
                    heappush(frontier, (new_cost, push_id, nxt))

        return []

    def _reconstruct(
        self, parent: Dict[State, Tuple[State, Action]], end: State
    ) -> List[Action]:
        actions: List[Action] = []
        current = end
        while current in parent:
            prev, act = parent[current]
            actions.append(act)
            current = prev
        actions.reverse()
        return actions


class AdaptiveGeometryDashAI:
    """Automatically tunes physics/search params until route is found."""

    def __init__(self, base_level: Level):
        self.base_level = base_level

    def _make_level(self, config: SolveConfig) -> Level:
        return Level(
            length=self.base_level.length,
            hazards=self.base_level.hazards,
            gravity=config.gravity,
            speed=config.speed,
            jump_velocity=config.jump_velocity,
        )

    def _candidate_configs(self) -> List[SolveConfig]:
        dt_values = (0.05, 0.04, 0.033)
        scale_values = (100, 120, 150)
        gravity_mult = (0.92, 1.0, 1.08)
        speed_mult = (0.94, 1.0, 1.06)
        jump_mult = (0.9, 1.0, 1.1, 1.2)

        configs: List[SolveConfig] = []
        for dt, scale, g_m, s_m, j_m in product(
            dt_values, scale_values, gravity_mult, speed_mult, jump_mult
        ):
            configs.append(
                SolveConfig(
                    dt=dt,
                    scale=scale,
                    gravity=self.base_level.gravity * g_m,
                    speed=self.base_level.speed * s_m,
                    jump_velocity=self.base_level.jump_velocity * j_m,
                )
            )
        return configs

    def solve(self, max_steps: int = 2200) -> Tuple[List[Action], Optional[SolveConfig]]:
        """Try multiple configs and return first successful route."""

        for config in self._candidate_configs():
            ai = GeometryDashAI(self._make_level(config), dt=config.dt, scale=config.scale)
            actions = ai.solve(max_steps=max_steps)
            if actions:
                return actions, config
        return [], None


def demo_level() -> Level:
    return Level(
        length=35.0,
        hazards=(
            Hazard(6.0, 7.0),
            Hazard(11.2, 12.3),
            Hazard(18.5, 20.4),
            Hazard(27.0, 28.0),
        ),
    )


def main() -> None:
    ai = AdaptiveGeometryDashAI(demo_level())
    actions, config = ai.solve()

    if not actions or config is None:
        print("Маршрут не найден даже после авто-подстройки")
        return

    jump_count = sum(1 for a in actions if a.press_jump)
    print("Маршрут найден с авто-подстройкой:")
    print(
        f"dt={config.dt}, scale={config.scale}, gravity={config.gravity:.2f}, "
        f"speed={config.speed:.2f}, jump_velocity={config.jump_velocity:.2f}"
    )
    print(f"Шагов: {len(actions)}")
    print(f"Прыжков: {jump_count}")
    print("Первые 40 действий:", " ".join(str(a) for a in actions[:40]))


if __name__ == "__main__":
    main()
