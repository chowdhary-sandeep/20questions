# MAPP - Multi-Agent Path Planning Environment

## Overview
- **Environment ID**: `MAPP`
- **Description**: A turn-based grid environment where multiple agents navigate simultaneously to assigned goals while avoiding obstacles and collisions.
- **Tags**: pathfinding, multi-agent, grid-world, reinforcement-learning, coordination

## Features
- Configurable grid sizes and obstacle densities
- Multiple agents with individual start and goal positions
- Turn-based movement with **5 possible actions per agent**
- Collision checking between agents and obstacles
- Reward system encouraging goal completion, safety, and efficiency

## Coordinate System and Actions
- Uses a **top-left origin coordinate system** where (0,0) is top-left of the grid
- Coordinates: x increases rightward, y increases downward
- Agent action codes and effects (5 actions):

| Action Code | Description | Movement (dx, dy) |
|-------------|-------------|-------------------|
| 0           | Move Left   | (-1, 0)           |
| 1           | Move Up     | (0, -1)           |
| 2           | Stay        | (0, 0)            |
| 3           | Move Down   | (0, 1)            |
| 4           | Move Right  | (1, 0)            |

## Evaluation Metrics
- **Goal completion rate**: Fraction of agents successfully reaching goals  
- **Collision count**: Number of collisions encountered  
- **Path efficiency**: Ratio of shortest path length to actual path taken  

## How to Run

Run evaluation with the repository's vf-eval tool (default settings):

```bash
uv run vf-eval MAPP
```

Example of a real evaluation run:

```bash
vf-eval MAPP -m gpt-4o --api-key-var LLM_API_KEY -n 4 -r 1 -t 1024 -T 0.5 -a '{"num_scenarios":10,"seed":42,"max_tries":50,"collision_verbose":true,"eval_filename":"mapp_data/eval_scenarios.json"}' --verbose
```


## Environment Parameters

| Parameter               | Type                | Default                          | Description                                            |
|-------------------------|---------------------|----------------------------------|--------------------------------------------------------|
| `num_scenarios`         | int                 | 128                              | Number of generated training scenarios                 |
| `grid_sizes`            | List[Tuple[int,int]]| [(5,5),(6,6)]                    | Candidate grid sizes used when generating scenarios    |
| `agent_counts`          | List[int]           | [2,3,4]                          | Candidate numbers of agents per scenario               |
| `obstacle_densities`    | List[float]         | [0.1,0.2,0.3]                    | Candidate obstacle densities used during generation    |
| `max_tries`             | int                 | 200                              | Maximum attempts to generate a valid scenario          |
| `max_steps`             | int                 | 32                               | Maximum steps per episode                              |
| `seed`                  | int                 | 42                               | Random seed for scenario generation                    |
| `scenario_verbose`      | bool                | False                            | If true, scenario generator prints verbose info        |
| `collision_verbose`     | bool                | False                            | If true, collision checks emit debug output            |
| `eval_filename`         | Optional[str]       | None                             | Optional path to a dataset of evaluation scenarios     |