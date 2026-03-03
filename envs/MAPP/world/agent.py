from dataclasses import dataclass
from typing import Tuple

@dataclass
class Agent:
    """Represents a single agent in the pathfinding environment."""
    
    id: int
    position: Tuple[int, int]
    goal: Tuple[int, int]

    # Coordinate system: origin (0,0) is top-left, x increases right, y increases down
    actions = {
        0: (-1, 0),   # LEFT (decrease x)
        1: (0, -1),   # UP (decrease y)
        2: (0, 0),    # STAY
        3: (0, 1),    # DOWN (increase y)
        4: (1, 0),    # RIGHT (increase x)

    }
    
    def distance_to_goal(self) -> float:
        """Manhattan distance to goal."""
        return abs(self.position[0] - self.goal[0]) + abs(self.position[1] - self.goal[1])
    
    def at_goal(self) -> bool:
        """Check if agent is at its goal position."""
        return self.position == self.goal
    
    def move(self, new_position: Tuple[int, int]):
        """Update agent position."""
        self.position = new_position