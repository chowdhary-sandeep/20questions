from typing import List, Tuple, Set, Optional
from .agent import Agent

class GridWorld:
    """Manages the grid environment, neighbors and collision detection."""
    
    def __init__(self,
                 scenario):
        self.width = scenario.width
        self.height = scenario.height
        self.obstacles = set(scenario.obstacles) if scenario.obstacles else set()
        self.agents = [Agent(id=i, position=pos, goal=goal) for i, (pos, goal) in enumerate(zip(scenario.agent_positions, scenario.goal_positions))]

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_free(self, pos: Tuple[int, int]) -> bool:
        """Check if position is free (no obstacle) and inside bounds."""
        if not self.is_valid_position(pos):
            return False
        return pos not in self.obstacles
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (4-connected, no diagonals)."""
        x, y = pos
        neighbors: List[Tuple[int, int]] = []
        # 4 directions only (no diagonals)
        directions = [(-1, 0),  # LEFT
                    (0, -1),  # UP
                    (0, 1),   # DOWN
                    (1, 0)]   # RIGHT

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Only add if destination is free (handles bounds and obstacles)
            if self.is_free((nx, ny)):
                neighbors.append((nx, ny))

        return neighbors


    def check_collision(self, actions: List[int], collision_verbose: bool = False) -> List[bool]:
        """Check collisions for the given actions for self.agents.
        Returns list of booleans per-agent indicating whether that agent
        is involved in a collision (environmental or agent-agent).
        Prints detailed reasons for each collision.
        
        Action mapping: 0=LEFT, 1=UP, 2=STAY, 3=DOWN, 4=RIGHT
        """
        n = len(self.agents)
        collisions = [False] * n
        curr = [tuple(a.position) for a in self.agents]
        intended: List[Tuple[int, int]] = [None] * n  # type: ignore
        collision_reasons = [""] * n

        # Compute intended positions
        for i, action in enumerate(actions):
            dx, dy = Agent.actions[action]
            cx, cy = curr[i]
            intended[i] = (cx + dx, cy + dy)

        # Pass 1: Environment collision checking
        for i, action in enumerate(actions):
            if action == 2:  # STAY never collides with environment
                continue

            nx, ny = intended[i]
            if not self.is_free((nx, ny)):
                collisions[i] = True
                if nx < 0 or ny < 0 or nx >= self.width or ny >= self.height:
                    collision_reasons[i] = f"OUT OF BOUNDS: Agent {i} tried to move to ({nx}, {ny}) which is outside grid bounds"
                else:
                    collision_reasons[i] = f"OBSTACLE: Agent {i} tried to move to ({nx}, {ny}) which contains an obstacle"

        # Pass 2: Agent-agent vertex conflicts
        pos_to_agents = {}
        for i, pos in enumerate(intended):
            pos_to_agents.setdefault(pos, []).append(i)

        for pos, agents_here in pos_to_agents.items():
            if len(agents_here) > 1:
                agent_list = ", ".join(map(str, agents_here))
                for idx in agents_here:
                    collisions[idx] = True
                    collision_reasons[idx] = f"VERTEX CONFLICT: Agents {agent_list} all trying to move to {pos}"

        # Pass 3: Swap (edge) conflicts
        for i in range(n):
            for j in range(i + 1, n):
                if intended[i] == curr[j] and intended[j] == curr[i]:
                    collisions[i] = True
                    collisions[j] = True
                    collision_reasons[i] = f"SWAP CONFLICT: Agent {i} at {curr[i]} and Agent {j} at {curr[j]} trying to swap positions"
                    collision_reasons[j] = f"SWAP CONFLICT: Agent {j} at {curr[j]} and Agent {i} at {curr[i]} trying to swap positions"

        # Pass 4: Check moves into currently occupied cells where occupant may collide and stay
        curr_pos_to_agent = {pos: i for i, pos in enumerate(curr)}

        for i, target in enumerate(intended):
            if collisions[i]:
                continue  # already colliding
            if target in curr_pos_to_agent:
                occupant = curr_pos_to_agent[target]
                occupant_intended = intended[occupant]
                occupant_stays = occupant_intended == curr[occupant]
                if occupant_stays or collisions[occupant]:
                    collisions[i] = True
                    collision_reasons[i] = f"OCCUPIED CELL CONFLICT: Agent {i} tries to move to {target} occupied by Agent {occupant} who stays due to collision"

        if collision_verbose:
            for i, reason in enumerate(collision_reasons):
                if collisions[i]:
                    action_name = {0: "LEFT", 1: "UP", 2: "STAY", 3: "DOWN", 4: "RIGHT"}[actions[i]]
                    print(f"Agent {i} COLLISION: Action {actions[i]} ({action_name}) - {reason}")

        return collisions




    def visualize_world(self) -> str:
        """
        Create ASCII visualization of scenario (top-left origin).
        Supports: agent and goal co-occupancy, no duplicate agent/goal positions, but agents can be on goals.
        """
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]

        # Mark obstacles
        for x, y in self.obstacles:
            grid[y][x] = '#'

        # Mark agents (A0, A1, etc.)
        for i, agent in enumerate(self.agents):
            x, y = agent.position
            grid[y][x] = f'A{i}'

        # Mark goals (G0, G1, etc.) from agents' goals
        for i, agent in enumerate(self.agents):
            x, y = agent.goal
            cell_val = grid[y][x]
            if cell_val == '.':
                grid[y][x] = f'G{i}'
            elif cell_val.startswith('A'):
                grid[y][x] = f'{cell_val}G{i}'  # Agent on goal, e.g., A1G0

        # Convert to string (top row first for proper visualization)
        result = []
        for row in grid:
            result.append(' '.join(f'{cell:>5}' for cell in row))

        return '\n'.join(result)