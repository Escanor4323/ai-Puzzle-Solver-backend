"""Maze generation, solving, difficulty calibration, and visibility.

Four classes — pure algorithms, no AI dependencies, no external services.
Playable the moment it's written.

Classes:
    MazeGenerator        — creates valid, solvable mazes
    MazeSolver           — A* pathfinding (perfect + companion)
    DifficultyCalibrator — Elo-based difficulty measurement & adjustment
    VisibilityEngine     — fog-of-war for MAZE_DARK
"""

from __future__ import annotations

import heapq
import random
from statistics import mean
from typing import Any

from data.models import (
    CellWalls,
    MazeCell,
    MazeGenerationParams,
    MazeRule,
    MazeState,
    PuzzleType,
)


# ── MazeGenerator ──────────────────────────────────────────────


class MazeGenerator:
    """Creates valid, solvable mazes with controllable properties."""

    def generate(self, params: MazeGenerationParams) -> MazeState:
        """Main entry point. Generate a complete, solvable maze.

        Selects algorithm based on maze_type when algorithm is "auto":
          - MAZE_DARK   -> recursive_backtracker (long corridors)
          - MAZE_CLASSIC -> wilsons (unbiased, uniform random)
          - MAZE_LOGIC  -> kruskals (controllable corridor/room balance)
        """
        algorithm = params.algorithm
        if algorithm == "auto":
            algorithm = {
                PuzzleType.MAZE_DARK: "recursive_backtracker",
                PuzzleType.MAZE_CLASSIC: "wilsons",
                PuzzleType.MAZE_LOGIC: "kruskals",
            }.get(params.maze_type, "wilsons")

        generator_fn = {
            "recursive_backtracker": self._recursive_backtracker,
            "wilsons": self._wilsons,
            "kruskals": self._kruskals,
        }[algorithm]

        grid = generator_fn(params.width, params.height)
        start, exit_ = self._place_start_exit(
            grid, params.width, params.height
        )

        rules: list[MazeRule] = []
        if params.maze_type == PuzzleType.MAZE_LOGIC:
            grid, rules = self._add_logic_elements(grid, params)

        # Compute optimal path length
        solver = MazeSolver()
        optimal_path = solver.solve(grid, start, exit_, rules=rules)
        if not optimal_path:
            # Fallback: regenerate if unsolvable (shouldn't happen
            # with correct algorithms, but safety net)
            return self.generate(params)

        maze = MazeState(
            grid=grid,
            width=params.width,
            height=params.height,
            start=start,
            exit=exit_,
            player_position=start,
            visited_cells=[start],
            rules=rules,
            optimal_path_length=len(optimal_path),
            difficulty_elo=params.target_elo,
        )

        return maze

    def _recursive_backtracker(
        self, width: int, height: int
    ) -> list[list[MazeCell]]:
        """DFS-based maze generation with directional bias.

        60% chance to continue in same direction for longer corridors.
        """
        grid = _make_grid(width, height)
        visited = [[False] * height for _ in range(width)]

        start_x = random.randint(0, width - 1)
        start_y = random.randint(0, height - 1)
        visited[start_x][start_y] = True

        stack: list[tuple[int, int, str | None]] = [
            (start_x, start_y, None)
        ]

        while stack:
            x, y, last_dir = stack[-1]
            neighbors = _unvisited_neighbors(
                x, y, width, height, visited
            )

            if not neighbors:
                stack.pop()
                continue

            # Directional bias: 60% chance to continue same direction
            chosen = None
            if last_dir and random.random() < 0.6:
                for nx, ny, d in neighbors:
                    if d == last_dir:
                        chosen = (nx, ny, d)
                        break

            if chosen is None:
                chosen = random.choice(neighbors)

            nx, ny, d = chosen
            _remove_wall_between(grid, x, y, nx, ny)
            visited[nx][ny] = True
            stack.append((nx, ny, d))

        return grid

    def _wilsons(
        self, width: int, height: int
    ) -> list[list[MazeCell]]:
        """Loop-erased random walk. Unbiased uniform spanning tree."""
        grid = _make_grid(width, height)
        all_cells = [
            (x, y) for x in range(width) for y in range(height)
        ]
        in_maze: set[tuple[int, int]] = set()

        # Start with one random cell in the maze
        first = random.choice(all_cells)
        in_maze.add(first)

        remaining = set(all_cells) - in_maze

        while remaining:
            # Pick a random cell not yet in the maze
            start = random.choice(list(remaining))
            path: list[tuple[int, int]] = [start]
            visited_in_walk: dict[tuple[int, int], int] = {
                start: 0
            }

            current = start
            while current not in in_maze:
                neighbors = _all_neighbors(
                    current[0], current[1], width, height
                )
                next_cell = random.choice(neighbors)

                if next_cell in visited_in_walk:
                    # Loop detected — erase the loop
                    loop_start = visited_in_walk[next_cell]
                    # Remove entries from visited_in_walk
                    for cell in path[loop_start + 1 :]:
                        del visited_in_walk[cell]
                    path = path[: loop_start + 1]
                else:
                    path.append(next_cell)
                    visited_in_walk[next_cell] = len(path) - 1

                current = next_cell

            # Add path to maze by carving walls
            for i in range(len(path) - 1):
                cx, cy = path[i]
                nx, ny = path[i + 1]
                _remove_wall_between(grid, cx, cy, nx, ny)
                in_maze.add(path[i])
            # The last cell in path connects to a cell already in maze
            last = path[-1]
            if last not in in_maze:
                _remove_wall_between(
                    grid, last[0], last[1], current[0], current[1]
                )
                in_maze.add(last)

            remaining = set(all_cells) - in_maze

        return grid

    def _kruskals(
        self, width: int, height: int
    ) -> list[list[MazeCell]]:
        """Kruskal's algorithm with weighted randomisation.

        Uses union-find. Random weights on walls control
        corridor vs room balance.
        """
        grid = _make_grid(width, height)

        # Union-Find
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        rank: dict[tuple[int, int], int] = {}

        for x in range(width):
            for y in range(height):
                parent[(x, y)] = (x, y)
                rank[(x, y)] = 0

        def find(cell: tuple[int, int]) -> tuple[int, int]:
            while parent[cell] != cell:
                parent[cell] = parent[parent[cell]]  # path compression
                cell = parent[cell]
            return cell

        def union(a: tuple[int, int], b: tuple[int, int]) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True

        # Collect all internal walls with random weights
        walls: list[tuple[float, int, int, int, int]] = []
        for x in range(width):
            for y in range(height):
                if x + 1 < width:
                    walls.append(
                        (random.random(), x, y, x + 1, y)
                    )
                if y + 1 < height:
                    walls.append(
                        (random.random(), x, y, x, y + 1)
                    )

        walls.sort()

        for _, x1, y1, x2, y2 in walls:
            if union((x1, y1), (x2, y2)):
                _remove_wall_between(grid, x1, y1, x2, y2)

        return grid

    def _add_logic_elements(
        self,
        grid: list[list[MazeCell]],
        params: MazeGenerationParams,
    ) -> tuple[list[list[MazeCell]], list[MazeRule]]:
        """Add logic elements to MAZE_LOGIC variant.

        Validates solvability after each addition.
        """
        rules: list[MazeRule] = []
        solver = MazeSolver()
        width = params.width
        height = params.height
        start, exit_ = _find_start_exit(grid, width, height)

        # Get the optimal path for placement guidance
        optimal_path = solver.solve(grid, start, exit_)
        if not optimal_path:
            return grid, rules

        # 1. Place colored tiles on/near the optimal path
        colors = ["red", "green", "blue"]
        num_colored = min(
            params.num_logic_rules, len(optimal_path) // 3
        )
        if num_colored > 0:
            # Place in the first 75% of path to ensure engagement
            placement_range = optimal_path[
                1 : max(2, int(len(optimal_path) * 0.75))
            ]
            chosen_positions = random.sample(
                placement_range, min(num_colored, len(placement_range))
            )
            for pos in chosen_positions:
                color = random.choice(colors)
                grid[pos[0]][pos[1]].color = color

            rules.append(
                MazeRule(
                    rule_type="colored_tile",
                    description="Colored tiles affect movement cost",
                    params={
                        "red": 3.0,
                        "green": 0.5,
                        "blue": 1.0,
                    },
                )
            )

        # 2. Place keys and locked doors
        if params.num_keys > 0:
            key_colors = colors[: params.num_keys]
            for kc in key_colors:
                # Key must be reachable before its door
                # Place key in first third, door in last third
                first_third = optimal_path[
                    1 : max(2, len(optimal_path) // 3)
                ]
                last_third = optimal_path[
                    max(1, 2 * len(optimal_path) // 3) : -1
                ]

                if first_third and last_third:
                    key_pos = random.choice(first_third)
                    door_pos = random.choice(last_third)

                    # Place key as an item in the cell
                    grid[key_pos[0]][key_pos[1]].color = kc

                    # Place locked door
                    grid[door_pos[0]][door_pos[1]].has_door = True
                    grid[door_pos[0]][door_pos[1]].door_color = kc

                    # Verify still solvable with rule-aware solver
                    test_rules = rules + [
                        MazeRule(
                            rule_type="locked_door",
                            description=f"A {kc} door blocks the path",
                            params={"color": kc},
                        )
                    ]
                    test_path = solver.solve(
                        grid, start, exit_, rules=test_rules
                    )
                    if test_path:
                        rules.append(test_rules[-1])
                    else:
                        # Revert if unsolvable
                        grid[door_pos[0]][
                            door_pos[1]
                        ].has_door = False
                        grid[door_pos[0]][
                            door_pos[1]
                        ].door_color = None

        # 3. Add teleporters (bidirectional)
        remaining_rules = params.num_logic_rules - len(rules)
        if remaining_rules > 0:
            # Find cells not on optimal path for teleporter placement
            non_path_cells = [
                (x, y)
                for x in range(width)
                for y in range(height)
                if (x, y) not in set(optimal_path)
                and (x, y) != start
                and (x, y) != exit_
            ]
            if len(non_path_cells) >= 2:
                tp_a, tp_b = random.sample(non_path_cells, 2)
                grid[tp_a[0]][tp_a[1]].is_teleporter = True
                grid[tp_a[0]][tp_a[1]].teleport_target = tp_b
                grid[tp_b[0]][tp_b[1]].is_teleporter = True
                grid[tp_b[0]][tp_b[1]].teleport_target = tp_a
                rules.append(
                    MazeRule(
                        rule_type="teleporter",
                        description="Teleporters transport you to a linked location",
                        params={
                            "a": list(tp_a),
                            "b": list(tp_b),
                        },
                    )
                )

        # 4. Add one-way passages if still room for rules
        remaining_rules = params.num_logic_rules - len(rules)
        if remaining_rules > 0 and len(optimal_path) > 4:
            # Pick a cell mid-path and make one direction one-way
            mid_idx = len(optimal_path) // 2
            mid_cell = optimal_path[mid_idx]
            next_cell = optimal_path[mid_idx + 1]
            # Determine the direction from mid to next
            dx = next_cell[0] - mid_cell[0]
            dy = next_cell[1] - mid_cell[1]
            direction = _delta_to_direction(dx, dy)
            if direction:
                grid[mid_cell[0]][
                    mid_cell[1]
                ].allowed_entry = [direction]
                rules.append(
                    MazeRule(
                        rule_type="one_way",
                        description="Some passages only allow travel in one direction",
                        params={
                            "cell": list(mid_cell),
                            "allowed": [direction],
                        },
                    )
                )
                # Verify solvability
                test_path = solver.solve(
                    grid, start, exit_, rules=rules
                )
                if not test_path:
                    # Revert
                    grid[mid_cell[0]][
                        mid_cell[1]
                    ].allowed_entry = []
                    rules.pop()

        return grid, rules

    def _place_start_exit(
        self,
        grid: list[list[MazeCell]],
        width: int,
        height: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Place start on left/top edge, exit on right/bottom edge.

        Constraint: Manhattan distance >= max(width, height) * 0.6
        """
        min_dist = int(max(width, height) * 0.6)

        for _ in range(100):  # max attempts
            # Start: left or top edge
            if random.random() < 0.5:
                start = (0, random.randint(0, height - 1))
            else:
                start = (random.randint(0, width - 1), 0)

            # Exit: right or bottom edge
            if random.random() < 0.5:
                exit_ = (
                    width - 1,
                    random.randint(0, height - 1),
                )
            else:
                exit_ = (
                    random.randint(0, width - 1),
                    height - 1,
                )

            dist = abs(start[0] - exit_[0]) + abs(
                start[1] - exit_[1]
            )
            if dist >= min_dist:
                return start, exit_

        # Fallback: corners
        return (0, 0), (width - 1, height - 1)


# ── MazeSolver ─────────────────────────────────────────────────


class MazeSolver:
    """A* pathfinding with multiple modes."""

    def solve(
        self,
        grid: list[list[MazeCell]],
        start: tuple[int, int],
        exit_: tuple[int, int],
        rules: list[MazeRule] | None = None,
    ) -> list[tuple[int, int]]:
        """Standard A* with Manhattan distance heuristic.

        Returns optimal path as list of (x, y) positions.
        Returns empty list if unsolvable.
        """
        return self._astar(grid, start, exit_, rules=rules)

    def solve_companion(
        self,
        grid: list[list[MazeCell]],
        start: tuple[int, int],
        exit_: tuple[int, int],
        epsilon: float = 0.0,
        rules: list[MazeRule] | None = None,
    ) -> list[tuple[int, int]]:
        """A* with epsilon-greedy noise injection.

        At each node expansion, with probability epsilon, pick a
        random valid neighbor instead of the f-score optimal one.
        """
        return self._astar(
            grid, start, exit_, epsilon=epsilon, rules=rules
        )

    def find_all_dead_ends(
        self, grid: list[list[MazeCell]]
    ) -> list[dict[str, Any]]:
        """Find all dead ends and measure their depth.

        A dead end is a cell with 3 walls (only one opening).
        Depth is distance back to nearest junction (cell with 2+ openings).
        """
        width = len(grid)
        height = len(grid[0]) if width > 0 else 0
        dead_ends: list[dict[str, Any]] = []

        for x in range(width):
            for y in range(height):
                cell = grid[x][y]
                openings = _count_openings(cell)
                if openings == 1:
                    # BFS back to nearest junction
                    depth = 0
                    cx, cy = x, y
                    visited = {(cx, cy)}
                    while True:
                        depth += 1
                        neighbors = _passable_neighbors(
                            grid, cx, cy, width, height
                        )
                        next_cells = [
                            n
                            for n in neighbors
                            if n not in visited
                        ]
                        if not next_cells:
                            break
                        cx, cy = next_cells[0]
                        visited.add((cx, cy))
                        if (
                            _count_openings(grid[cx][cy]) >= 3
                        ):
                            break

                    dead_ends.append(
                        {
                            "position": (x, y),
                            "depth": depth,
                            "junction": (cx, cy),
                        }
                    )

        return dead_ends

    def find_alternative_paths(
        self,
        grid: list[list[MazeCell]],
        start: tuple[int, int],
        exit_: tuple[int, int],
        n: int = 5,
    ) -> list[list[tuple[int, int]]]:
        """Find n alternative paths by penalising previously found edges.

        After each A* run, increase cost of edges on the found path
        by 2x, then re-solve.
        """
        paths: list[list[tuple[int, int]]] = []
        # Track edge penalties: (from, to) -> cost multiplier
        penalties: dict[
            tuple[tuple[int, int], tuple[int, int]], float
        ] = {}

        for _ in range(n):
            path = self._astar(
                grid, start, exit_, penalties=penalties
            )
            if not path:
                break
            paths.append(path)

            # Penalise edges on this path
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                penalties[edge] = penalties.get(edge, 1.0) * 2.0
                # Penalise reverse edge too
                rev = (path[i + 1], path[i])
                penalties[rev] = penalties.get(rev, 1.0) * 2.0

        return paths

    def _astar(
        self,
        grid: list[list[MazeCell]],
        start: tuple[int, int],
        exit_: tuple[int, int],
        epsilon: float = 0.0,
        rules: list[MazeRule] | None = None,
        penalties: dict[
            tuple[tuple[int, int], tuple[int, int]], float
        ]
        | None = None,
        player_state: dict[str, Any] | None = None,
    ) -> list[tuple[int, int]]:
        """Core A* implementation.

        Parameters
        ----------
        epsilon : float
            Probability of choosing a random neighbor (companion mode).
        rules : list[MazeRule] | None
            Logic rules affecting edge costs.
        penalties : dict | None
            Edge cost multipliers for alternative path finding.
        """
        width = len(grid)
        height = len(grid[0]) if width > 0 else 0

        # For rule-aware solving, track a simulated player state
        # to handle keys and doors correctly
        sim_state = player_state or {"keys": []}

        # Priority queue: (f_score, counter, (x, y), keys_snapshot)
        counter = 0
        open_set: list[
            tuple[float, int, tuple[int, int], tuple[str, ...]]
        ] = []
        keys_tuple = tuple(sorted(sim_state.get("keys", [])))
        heapq.heappush(
            open_set,
            (self._heuristic(start, exit_), counter, start, keys_tuple),
        )

        # g_score: (position, keys_state) -> cost
        g_score: dict[
            tuple[tuple[int, int], tuple[str, ...]], float
        ] = {(start, keys_tuple): 0.0}

        # came_from for path reconstruction
        came_from: dict[
            tuple[tuple[int, int], tuple[str, ...]],
            tuple[tuple[int, int], tuple[str, ...]],
        ] = {}

        visited: set[
            tuple[tuple[int, int], tuple[str, ...]]
        ] = set()

        while open_set:
            f, _, current, current_keys = heapq.heappop(
                open_set
            )
            state_key = (current, current_keys)

            if current == exit_:
                # Reconstruct path
                path = [current]
                sk = state_key
                while sk in came_from:
                    sk = came_from[sk]
                    path.append(sk[0])
                path.reverse()
                return path

            if state_key in visited:
                continue
            visited.add(state_key)

            neighbors = _passable_neighbors(
                grid, current[0], current[1], width, height
            )

            # Epsilon-greedy: sometimes pick random neighbor
            if epsilon > 0 and random.random() < epsilon and neighbors:
                # Still process all neighbors but randomize order
                random.shuffle(neighbors)

            for nx, ny in neighbors:
                neighbor = (nx, ny)
                neighbor_cell = grid[nx][ny]

                # Compute edge cost
                new_keys = list(current_keys)
                edge_cost = self._rule_aware_cost(
                    grid[current[0]][current[1]],
                    neighbor_cell,
                    new_keys,
                    rules,
                )

                if edge_cost == float("inf"):
                    continue

                # Apply penalties for alternative path finding
                if penalties:
                    edge = (current, neighbor)
                    edge_cost *= penalties.get(edge, 1.0)

                # Collect keys at neighbor
                if (
                    neighbor_cell.color
                    and neighbor_cell.color
                    not in new_keys
                    and not neighbor_cell.has_door
                ):
                    # If it's a colored tile that acts as a key
                    # (in logic mode, colored cells near doors are keys)
                    for rule in (rules or []):
                        if (
                            rule.rule_type == "locked_door"
                            and rule.params.get("color")
                            == neighbor_cell.color
                        ):
                            new_keys.append(neighbor_cell.color)
                            break

                new_keys_tuple = tuple(sorted(new_keys))
                new_state_key = (neighbor, new_keys_tuple)
                tentative_g = (
                    g_score.get(state_key, float("inf"))
                    + edge_cost
                )

                if tentative_g < g_score.get(
                    new_state_key, float("inf")
                ):
                    g_score[new_state_key] = tentative_g
                    f_score = tentative_g + self._heuristic(
                        neighbor, exit_
                    )
                    came_from[new_state_key] = state_key
                    counter += 1
                    heapq.heappush(
                        open_set,
                        (
                            f_score,
                            counter,
                            neighbor,
                            new_keys_tuple,
                        ),
                    )

        return []  # No path found

    @staticmethod
    def _heuristic(
        a: tuple[int, int], b: tuple[int, int]
    ) -> float:
        """Manhattan distance — admissible on a 4-connected grid."""
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    @staticmethod
    def _rule_aware_cost(
        current_cell: MazeCell,
        neighbor_cell: MazeCell,
        player_keys: list[str],
        rules: list[MazeRule] | None,
    ) -> float:
        """Compute edge cost considering logic rules.

        Base cost: 1.0
        Modifications per rule type:
          - colored_tile red: 3.0
          - colored_tile green: 0.5
          - locked_door without key: inf
          - locked_door with key: 1.0
          - one_way wrong direction: inf
          - teleporter: 0.5
        """
        cost = 1.0

        if not rules:
            return cost

        for rule in rules:
            if rule.rule_type == "colored_tile":
                if neighbor_cell.color in rule.params:
                    cost = rule.params[neighbor_cell.color]

            elif rule.rule_type == "locked_door":
                if (
                    neighbor_cell.has_door
                    and neighbor_cell.door_color
                ):
                    if (
                        neighbor_cell.door_color
                        not in player_keys
                    ):
                        return float("inf")
                    # If player has key, consume it
                    if (
                        neighbor_cell.door_color in player_keys
                    ):
                        player_keys.remove(
                            neighbor_cell.door_color
                        )

            elif rule.rule_type == "one_way":
                if neighbor_cell.allowed_entry:
                    # Check if we're entering from an allowed direction
                    dx = (
                        neighbor_cell.x - current_cell.x
                    )
                    dy = (
                        neighbor_cell.y - current_cell.y
                    )
                    entry_dir = _delta_to_direction(dx, dy)
                    if (
                        entry_dir
                        and entry_dir
                        not in neighbor_cell.allowed_entry
                    ):
                        return float("inf")

            elif rule.rule_type == "teleporter":
                if neighbor_cell.is_teleporter:
                    cost = 0.5

        return cost


# ── DifficultyCalibrator ───────────────────────────────────────


class DifficultyCalibrator:
    """Measures maze difficulty and adjusts to hit Elo targets."""

    def __init__(
        self, solver: MazeSolver | None = None
    ) -> None:
        self.solver = solver or MazeSolver()

    def compute_difficulty(
        self,
        maze: MazeState,
        optimal_path: list[tuple[int, int]],
    ) -> int:
        """Compute difficulty Elo from maze properties.

        Formula combines:
          - exploration_ratio (how much of the maze is off-path)
          - decision_density (junctions on optimal path)
          - max_dead_end_depth (deepest trap)
          - ambiguity (how similar alternative paths are)

        Maps raw 0.0-1.0 score to 800-2000 Elo range.
        """
        total_cells = maze.width * maze.height
        path_len = len(optimal_path)

        if path_len == 0 or total_cells == 0:
            return 800

        # Exploration ratio
        exploration_ratio = 1.0 - (path_len / total_cells)

        # Decision density: cells on optimal path with 3+ openings
        decision_points = 0
        for pos in optimal_path:
            cell = maze.grid[pos[0]][pos[1]]
            if _count_openings(cell) >= 3:
                decision_points += 1
        decision_density = decision_points / path_len

        # Dead end depth
        dead_ends = self.solver.find_all_dead_ends(maze.grid)
        if dead_ends:
            max_depth = max(d["depth"] for d in dead_ends)
            max_dead_end_depth = min(
                max_depth / path_len, 1.0
            )
        else:
            max_dead_end_depth = 0.0

        # Ambiguity: how different are alternative paths
        alt_paths = self.solver.find_alternative_paths(
            maze.grid, maze.start, maze.exit, n=5
        )
        if alt_paths:
            overlaps = []
            opt_set = set(optimal_path)
            for alt in alt_paths:
                alt_set = set(alt)
                if opt_set:
                    overlap = len(opt_set & alt_set) / len(
                        opt_set
                    )
                    overlaps.append(overlap)
            avg_overlap = mean(overlaps) if overlaps else 1.0
            ambiguity = 1.0 - avg_overlap
        else:
            ambiguity = 0.0

        raw = (
            exploration_ratio * 0.25
            + decision_density * 0.30
            + max_dead_end_depth * 0.25
            + ambiguity * 0.20
        )

        # Clamp raw to [0, 1]
        raw = max(0.0, min(1.0, raw))

        elo = int(800 + raw * 1200)
        return elo

    def adjust_to_target(
        self,
        maze: MazeState,
        target_elo: int,
        tolerance: int = 100,
        max_iterations: int = 20,
    ) -> MazeState:
        """Iteratively adjust maze difficulty toward target Elo.

        Modifies the maze in-place until difficulty is within
        tolerance of target, or max_iterations reached.
        """
        best_maze = maze
        best_diff = float("inf")

        for _ in range(max_iterations):
            path = self.solver.solve(
                maze.grid, maze.start, maze.exit, rules=maze.rules
            )
            if not path:
                break

            current_elo = self.compute_difficulty(maze, path)
            diff = abs(current_elo - target_elo)

            if diff < best_diff:
                best_diff = diff
                best_maze = maze.model_copy(deep=True)
                best_maze.optimal_path_length = len(path)
                best_maze.difficulty_elo = current_elo

            if diff <= tolerance:
                maze.optimal_path_length = len(path)
                maze.difficulty_elo = current_elo
                return maze

            if current_elo < target_elo:
                self._increase_difficulty(maze)
            else:
                self._decrease_difficulty(maze)

        return best_maze

    def _increase_difficulty(self, maze: MazeState) -> None:
        """Make the maze harder. Pick one at random:
        - Extend deepest dead end
        - Add a fork on the optimal path
        - Remove a shortcut
        """
        action = random.choice(
            ["extend_dead_end", "add_fork", "remove_shortcut"]
        )

        if action == "extend_dead_end":
            dead_ends = self.solver.find_all_dead_ends(
                maze.grid
            )
            if dead_ends:
                deepest = max(
                    dead_ends, key=lambda d: d["depth"]
                )
                self._extend_dead_end(
                    maze.grid,
                    deepest["position"],
                    maze.width,
                    maze.height,
                    random.randint(2, 4),
                )

        elif action == "add_fork":
            path = self.solver.solve(
                maze.grid, maze.start, maze.exit
            )
            if path and len(path) > 2:
                # Pick a point on the optimal path
                idx = random.randint(1, len(path) - 2)
                fork_point = path[idx]
                self._add_dead_end_branch(
                    maze.grid,
                    fork_point,
                    maze.width,
                    maze.height,
                )

        elif action == "remove_shortcut":
            # Add a wall between two adjacent cells that aren't
            # on the optimal path, if removing it doesn't break
            # solvability
            path = self.solver.solve(
                maze.grid, maze.start, maze.exit
            )
            path_set = set(path) if path else set()

            for _ in range(20):
                x = random.randint(0, maze.width - 2)
                y = random.randint(0, maze.height - 1)
                if (x, y) not in path_set and (
                    x + 1,
                    y,
                ) not in path_set:
                    cell = maze.grid[x][y]
                    if not cell.walls.east:
                        # Try adding wall
                        cell.walls.east = True
                        maze.grid[x + 1][
                            y
                        ].walls.west = True

                        # Verify solvability
                        test = self.solver.solve(
                            maze.grid,
                            maze.start,
                            maze.exit,
                        )
                        if test:
                            break
                        # Revert
                        cell.walls.east = False
                        maze.grid[x + 1][
                            y
                        ].walls.west = False

    def _decrease_difficulty(self, maze: MazeState) -> None:
        """Make the maze easier. Pick one at random:
        - Collapse deepest dead end
        - Remove a fork near exit
        - Carve a shortcut
        """
        action = random.choice(
            ["collapse_dead_end", "remove_fork", "carve_shortcut"]
        )

        if action == "collapse_dead_end":
            dead_ends = self.solver.find_all_dead_ends(
                maze.grid
            )
            if dead_ends:
                deepest = max(
                    dead_ends, key=lambda d: d["depth"]
                )
                # Fill the dead end by adding walls
                pos = deepest["position"]
                cell = maze.grid[pos[0]][pos[1]]
                cell.walls = CellWalls(
                    north=True,
                    south=True,
                    east=True,
                    west=True,
                )
                # Also wall off neighbors pointing to this cell
                for nx, ny, d in [
                    (pos[0], pos[1] - 1, "south"),
                    (pos[0], pos[1] + 1, "north"),
                    (pos[0] - 1, pos[1], "east"),
                    (pos[0] + 1, pos[1], "west"),
                ]:
                    if (
                        0 <= nx < maze.width
                        and 0 <= ny < maze.height
                    ):
                        setattr(
                            maze.grid[nx][ny].walls, d, True
                        )

        elif action == "carve_shortcut":
            # Find two cells on the optimal path and carve
            # a connection between adjacent off-path cells
            path = self.solver.solve(
                maze.grid, maze.start, maze.exit
            )
            if path and len(path) > 6:
                # Pick two points separated by some distance
                i = len(path) // 3
                j = 2 * len(path) // 3
                # Try to carve between cells near these points
                for _ in range(10):
                    x1, y1 = path[i]
                    x2, y2 = path[j]
                    # Find adjacent cells to carve through
                    if (
                        abs(x1 - x2) <= 1
                        and abs(y1 - y2) <= 1
                    ):
                        _remove_wall_between(
                            maze.grid, x1, y1, x2, y2
                        )
                        break
                    # Move points closer
                    i += 1
                    j -= 1
                    if i >= j:
                        break

    def _extend_dead_end(
        self,
        grid: list[list[MazeCell]],
        pos: tuple[int, int],
        width: int,
        height: int,
        length: int,
    ) -> None:
        """Extend a dead end by carving further."""
        x, y = pos
        for _ in range(length):
            neighbors = [
                (nx, ny)
                for nx, ny in [
                    (x - 1, y),
                    (x + 1, y),
                    (x, y - 1),
                    (x, y + 1),
                ]
                if 0 <= nx < width
                and 0 <= ny < height
                and _count_openings(grid[nx][ny]) <= 1
            ]
            if not neighbors:
                break
            nx, ny = random.choice(neighbors)
            _remove_wall_between(grid, x, y, nx, ny)
            x, y = nx, ny

    def _add_dead_end_branch(
        self,
        grid: list[list[MazeCell]],
        pos: tuple[int, int],
        width: int,
        height: int,
    ) -> None:
        """Add a branching dead-end from a position."""
        x, y = pos
        # Find a walled neighbor to branch into
        for nx, ny in [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
        ]:
            if (
                0 <= nx < width
                and 0 <= ny < height
                and _count_openings(grid[nx][ny]) == 0
            ):
                _remove_wall_between(grid, x, y, nx, ny)
                # Extend the branch a bit
                self._extend_dead_end(
                    grid, (nx, ny), width, height, 2
                )
                break


# ── VisibilityEngine ───────────────────────────────────────────


class VisibilityEngine:
    """Fog-of-war visibility for MAZE_DARK."""

    def get_visible_cells(
        self,
        grid: list[list[MazeCell]],
        position: tuple[int, int],
        visited: set[tuple[int, int]] | list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Return cells the player can currently see.

        - Current cell (always visible)
        - Adjacent cells reachable without a wall between them
        - All previously visited cells (memory)
        """
        if isinstance(visited, list):
            visited_set = set(
                tuple(v) for v in visited
            )
        else:
            visited_set = visited

        visible = set(visited_set)
        visible.add(position)

        width = len(grid)
        height = len(grid[0]) if width > 0 else 0

        # Add passable neighbors
        neighbors = _passable_neighbors(
            grid,
            position[0],
            position[1],
            width,
            height,
        )
        for n in neighbors:
            visible.add(n)

        return list(visible)

    def get_description_context(
        self,
        grid: list[list[MazeCell]],
        position: tuple[int, int],
    ) -> dict[str, Any]:
        """Return context dict for AI to describe surroundings.

        The AI uses this to narrate without seeing the full grid.
        """
        x, y = position
        width = len(grid)
        height = len(grid[0]) if width > 0 else 0
        cell = grid[x][y]

        available_exits: list[str] = []
        walls: list[str] = []
        corridor_lengths: dict[str, int] = {}
        dead_end_ahead: dict[str, bool] = {}
        items_visible: list[str] = []

        directions = {
            "north": (0, -1, "north"),
            "south": (0, 1, "south"),
            "east": (1, 0, "east"),
            "west": (-1, 0, "west"),
        }

        for dir_name, (dx, dy, wall_name) in directions.items():
            has_wall = getattr(cell.walls, wall_name)
            if has_wall:
                walls.append(dir_name)
            else:
                available_exits.append(dir_name)
                # Measure corridor length until fork or dead end
                length = 0
                cx, cy = x + dx, y + dy
                while (
                    0 <= cx < width
                    and 0 <= cy < height
                ):
                    length += 1
                    ncell = grid[cx][cy]
                    openings = _count_openings(ncell)
                    if openings != 2:
                        # Fork or dead end
                        dead_end_ahead[dir_name] = (
                            openings == 1
                        )
                        break
                    cx += dx
                    cy += dy
                corridor_lengths[dir_name] = length

                # Check for items in adjacent cell
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    adj = grid[nx][ny]
                    if adj.color:
                        items_visible.append(
                            f"{adj.color} tile"
                        )
                    if adj.has_door:
                        items_visible.append(
                            f"{adj.door_color} door"
                        )
                    if adj.is_teleporter:
                        items_visible.append("teleporter")

        # Check proximity to exit
        solver = MazeSolver()
        # Simple Manhattan distance check for "near exit"
        near_exit = False
        optimal = solver.solve(
            grid, position, _find_exit(grid, width, height)
        )
        if optimal:
            near_exit = len(optimal) <= max(
                3, int(len(optimal) * 0.2)
            )

        return {
            "available_exits": available_exits,
            "walls": walls,
            "corridor_lengths": corridor_lengths,
            "dead_end_ahead": dead_end_ahead,
            "near_exit": near_exit,
            "items_visible": items_visible,
        }


# ── Helper Functions ───────────────────────────────────────────


def _make_grid(
    width: int, height: int
) -> list[list[MazeCell]]:
    """Create a grid of fully-walled cells."""
    return [
        [
            MazeCell(
                x=x,
                y=y,
                walls=CellWalls(
                    north=True,
                    south=True,
                    east=True,
                    west=True,
                ),
            )
            for y in range(height)
        ]
        for x in range(width)
    ]


def _remove_wall_between(
    grid: list[list[MazeCell]],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> None:
    """Remove the wall between two adjacent cells."""
    dx, dy = x2 - x1, y2 - y1
    if dx == 1:
        grid[x1][y1].walls.east = False
        grid[x2][y2].walls.west = False
    elif dx == -1:
        grid[x1][y1].walls.west = False
        grid[x2][y2].walls.east = False
    elif dy == 1:
        grid[x1][y1].walls.south = False
        grid[x2][y2].walls.north = False
    elif dy == -1:
        grid[x1][y1].walls.north = False
        grid[x2][y2].walls.south = False


def _unvisited_neighbors(
    x: int,
    y: int,
    width: int,
    height: int,
    visited: list[list[bool]],
) -> list[tuple[int, int, str]]:
    """Return unvisited neighbors with direction labels."""
    result: list[tuple[int, int, str]] = []
    for dx, dy, d in [
        (0, -1, "north"),
        (0, 1, "south"),
        (1, 0, "east"),
        (-1, 0, "west"),
    ]:
        nx, ny = x + dx, y + dy
        if (
            0 <= nx < width
            and 0 <= ny < height
            and not visited[nx][ny]
        ):
            result.append((nx, ny, d))
    return result


def _all_neighbors(
    x: int, y: int, width: int, height: int
) -> list[tuple[int, int]]:
    """Return all valid neighbor coordinates."""
    result: list[tuple[int, int]] = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            result.append((nx, ny))
    return result


def _passable_neighbors(
    grid: list[list[MazeCell]],
    x: int,
    y: int,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    """Return neighbors reachable without passing through a wall."""
    cell = grid[x][y]
    result: list[tuple[int, int]] = []
    for dx, dy, wall in [
        (0, -1, "north"),
        (0, 1, "south"),
        (1, 0, "east"),
        (-1, 0, "west"),
    ]:
        nx, ny = x + dx, y + dy
        if (
            0 <= nx < width
            and 0 <= ny < height
            and not getattr(cell.walls, wall)
        ):
            result.append((nx, ny))
    return result


def _count_openings(cell: MazeCell) -> int:
    """Count how many sides of a cell have no wall."""
    return sum(
        1
        for w in [
            cell.walls.north,
            cell.walls.south,
            cell.walls.east,
            cell.walls.west,
        ]
        if not w
    )


def _delta_to_direction(
    dx: int, dy: int
) -> str | None:
    """Convert a delta to a direction string."""
    return {
        (0, -1): "north",
        (0, 1): "south",
        (1, 0): "east",
        (-1, 0): "west",
    }.get((dx, dy))


def _find_start_exit(
    grid: list[list[MazeCell]],
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find start and exit from an existing grid (corners fallback)."""
    return (0, 0), (width - 1, height - 1)


def _find_exit(
    grid: list[list[MazeCell]],
    width: int,
    height: int,
) -> tuple[int, int]:
    """Find exit position (bottom-right corner fallback)."""
    return (width - 1, height - 1)
