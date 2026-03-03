"""
Comprehensive dataset generator for multi-agent pathfinding scenarios.
Uses top-left origin coordinate system where (0,0) is top-left, 
x increases rightward, and y increases downward.
"""

import re
import json
import random
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ScenarioConfig:
    """Configuration for a single pathfinding scenario."""
    scenario_id: str
    width: int
    height: int
    agent_positions: List[Tuple[int, int]]
    goal_positions: List[Tuple[int, int]]
    obstacles: List[Tuple[int, int]]
    max_steps: int
    
    def to_dict(self) -> Dict[str, Any]:
        def coords_to_str(coords):
            return [f"[{x},{y}]" for x, y in coords]
        return {
            'scenario_id': self.scenario_id,
            'width': self.width,
            'height': self.height,
            'agent_positions': coords_to_str(self.agent_positions),
            'goal_positions': coords_to_str(self.goal_positions),
            'obstacles': coords_to_str(self.obstacles),
            'max_steps': self.max_steps
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        def str_to_tuple(s):
            numbers = re.findall(r'\d+', s)
            return (int(numbers[0]), int(numbers[1]))
        agent_positions = [str_to_tuple(pos) for pos in data['agent_positions']]
        goal_positions = [str_to_tuple(pos) for pos in data['goal_positions']]
        obstacles = [str_to_tuple(pos) for pos in data['obstacles']]
        return cls(
            scenario_id=data['scenario_id'],
            width=data['width'],
            height=data['height'],
            agent_positions=agent_positions,
            goal_positions=goal_positions,
            obstacles=obstacles,
            max_steps=data['max_steps']
        )


class ScenarioGenerator:
    """
    Generates diverse multi-agent pathfinding scenarios for 4-directional movement.
    Actions: 1=LEFT, 2=UP, 3=STAY, 4=DOWN, 5=RIGHT
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def validate_scenario(self, config: ScenarioConfig) -> bool:
        """
        Validates full scenario integrity for grid-based multi-agent environments.
        Returns True if the scenario is valid, otherwise False.
        """
        agent_positions = getattr(config, "agent_positions", None)
        goal_positions = getattr(config, "goal_positions", None)
        width = getattr(config, "width", None)
        height = getattr(config, "height", None)
        obstacles = set(getattr(config, "obstacles", []) or [])

        # Check required fields
        if agent_positions is None or goal_positions is None:
            return False
        if width is None or height is None:
            return False
        if len(agent_positions) == 0 or len(goal_positions) == 0:
            return False
        if len(agent_positions) != len(goal_positions):
            return False

        # Helper for in-bounds
        def in_bounds(pos):
            x, y = pos
            return 0 <= x < width and 0 <= y < height

        # Validate agent positions (bounds + obstacle overlap + uniqueness)
        if any(not in_bounds(pos) or pos in obstacles for pos in agent_positions):
            return False
        if len(agent_positions) != len(set(agent_positions)):
            return False

        # Validate goal positions (bounds + obstacle overlap + uniqueness)
        if any(not in_bounds(pos) or pos in obstacles for pos in goal_positions):
            return False
        if len(goal_positions) != len(set(goal_positions)):
            return False

        # Validate obstacles (all inside grid)
        if any(not in_bounds(pos) for pos in obstacles):
            return False

        return True
            
    def _is_solvable(self, config: ScenarioConfig) -> bool:
        """
        Check if all agents can reach their goals using 4-directional pathfinding.
        Updated to use only LEFT, UP, DOWN, RIGHT movements (no diagonals).
        """
        obstacle_set = set(config.obstacles)
        width, height = config.width, config.height
        
        # 4-directional movement only (matching your new action system)
        directions = [
            (-1, 0),  # LEFT (action 0)
            (0, -1),  # UP (action 1)
            (0, 1),   # DOWN (action 3)
            (1, 0),   # RIGHT (action 4)
            # Note: STAY (action 2) is (0, 0) but not needed for pathfinding
        ]

        def get_neighbors(x, y):
            """Get valid 4-connected neighbors (no diagonal moves)."""
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if (nx, ny) not in obstacle_set:
                        yield (nx, ny)

        def can_reach_goal(start, goal):
            """BFS pathfinding to check reachability with 4-directional movement."""
            if start == goal:
                return True
                
            queue = deque([start])
            visited = {start}
            
            while queue:
                x, y = queue.popleft()
                
                for nx, ny in get_neighbors(x, y):
                    if (nx, ny) == goal:
                        return True
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
            
            return False

        # Check if all agents can reach their respective goals
        return all(can_reach_goal(start, goal) 
                  for start, goal in zip(config.agent_positions, config.goal_positions))

    def generate_random_scenario(self, width: int, height: int, num_agents: int,
                                obstacle_density: float = 0.2, max_steps: Optional[int] = None) -> ScenarioConfig:
        """Generate random scenario with given obstacle density and max_steps."""
        
        scenario_id = f"random_{width}x{height}_{num_agents}_{random.randint(1, 9999)}"
        
        all_positions = [(x, y) for x in range(width) for y in range(height)]
        random.shuffle(all_positions)
        
        total_cells = width * height
        num_obstacles = int(total_cells * obstacle_density)
        
        # Initially assign obstacles
        obstacles = all_positions[:num_obstacles]
        free_positions = all_positions[num_obstacles:]
        
        required_space = 2 * num_agents  # Agents + goals
        
        # Raise error if free space is insufficient for unique agent and goal placement
        if len(free_positions) < required_space:
            raise ValueError(
                f"Not enough free space to place {num_agents} agents and goals uniquely "
                f"with obstacle density {obstacle_density}. Need {required_space} free cells, "
                f"but only {len(free_positions)} available."
            )
        
        # Randomly select agent and goal positions from free space, without overlap
        selected_positions = random.sample(free_positions, required_space)
        agent_positions = selected_positions[:num_agents]
        goal_positions = selected_positions[num_agents:]
        
        # Set max_steps: use given or default based on Manhattan distance heuristic
        if max_steps is None:
            # For 4-directional movement, use Manhattan distance * 2 as heuristic
            max_manhattan = max(abs(ax - gx) + abs(ay - gy) 
                               for (ax, ay), (gx, gy) in zip(agent_positions, goal_positions))
            max_steps = max(total_cells // 2, max_manhattan * 3)  # Conservative estimate
        
        return ScenarioConfig(
            scenario_id=scenario_id,
            width=width,
            height=height,
            agent_positions=agent_positions,
            goal_positions=goal_positions,
            obstacles=obstacles,
            max_steps=max_steps,
        )
    
    def generate_dataset(self, 
                        num_scenarios: int = 128,
                        grid_sizes: List[Tuple[int, int]] = None,
                        agent_counts: List[int] = None,
                        obstacle_densities: List[float] = None,
                        max_steps: Optional[int] = None,
                        max_tries: Optional[int] = None,
                        scenario_verbose: bool = False) -> List[ScenarioConfig]:
        """Generate a complete dataset of pathfinding scenarios."""

        if grid_sizes is None:
            grid_sizes = [(5, 5), (8, 8), (10, 10), (12, 12)] 
        if agent_counts is None:
            agent_counts = [2, 3, 4]
        if obstacle_densities is None:
            obstacle_densities = [0.1, 0.2, 0.3]

        scenarios = []
        max_tries = max_tries or num_scenarios * 10
        attempts = 0

        print(f"Generating {num_scenarios} scenarios with 4-directional movement...")

        while len(scenarios) < num_scenarios and attempts < max_tries:
            attempts += 1
            width, height = random.choice(grid_sizes)
            num_agents = random.choice(agent_counts)
            obstacle_density = random.choice(obstacle_densities)

            try:
                config = self.generate_random_scenario(
                    width, height, num_agents, obstacle_density, max_steps
                )
                
                if self.validate_scenario(config) and self._is_solvable(config):
                    scenarios.append(config)
                    if scenario_verbose and len(scenarios) % 10 == 0:
                        print(f"Generated {len(scenarios)}/{num_scenarios} scenarios...")
                        
            except Exception as e:
                if scenario_verbose:
                    print(f"Attempt {attempts} failed: {e}")
                continue

        if len(scenarios) < num_scenarios:
            print(f"Warning: Only generated {len(scenarios)} solvable scenarios after {attempts} attempts.")
        else:
            print(f"Successfully generated {len(scenarios)} scenarios after {attempts} attempts.")

        return scenarios
    
    def save_dataset(self, scenarios: List[ScenarioConfig], filename: str):
        """Save dataset of scenarios to a JSON file."""
        data = {
            'scenarios': [scenario.to_dict() for scenario in scenarios],
            'metadata': {
                'total_scenarios': len(scenarios),
                'coordinate_system': 'top-left origin, x increases right, y increases down'
            }
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load_dataset(self, filename: str) -> List[ScenarioConfig]:
        """Load dataset of scenarios from a JSON file."""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [ScenarioConfig.from_dict(s) for s in data['scenarios']]


