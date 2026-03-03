import json
import re
from dataclasses import dataclass, asdict
from typing import List, Any, Tuple, Dict, Optional

import verifiers as vf
from datasets import Dataset
from verifiers.types import ChatMessage, Messages, State

from world.scenarioGenerator import ScenarioGenerator, ScenarioConfig
from world.gridworld import GridWorld
from world.agent import Agent


class MappEnv(vf.MultiTurnEnv):
    """Multi-agent planning environment."""

    def __init__(self, dataset: Dataset, eval_dataset: Dataset, **kwargs):
        rubric = self._make_rubric()
        super().__init__(dataset=dataset, eval_dataset=eval_dataset, rubric=rubric, **kwargs)


    async def env_response(self, messages: Messages, state: State, **kwargs) -> Tuple[List[ChatMessage], State]:

        # Find last assistant message with actions
        last_assistant = None
        for m in reversed(messages):
            if m.get("role") == "assistant":
                last_assistant = m.get("content")
                break

        if last_assistant is None:
            raise ValueError("No assistant message found in messages")

        scenario = self._build_scenario_from_game(state)
        gw = GridWorld(scenario)

        # Parse actions from assistant output
        num_agents = len(gw.agents)
        actions = self.extract_actions(last_assistant, num_agents)

        # Pad actions with STAY as needed or truncate
        while len(actions) < num_agents:
            actions.append(2)
        actions = actions[:num_agents]

        collision_flags = gw.check_collision(actions, collision_verbose=kwargs.get("collision_verbose"))

        new_positions = []
        for i, a in enumerate(actions):
            if collision_flags[i]:
                new_positions.append(list(gw.agents[i].position))
            else:
                dx, dy = Agent.actions[a]
                x,y = gw.agents[i].position
                new_positions.append([x+dx,y+dy])

        new_state = state.copy()

        new_state["info"]["step_number"] += 1
        new_state["info"]["current_agent_positions"] = new_positions
        new_state["info"]["action_history"][str(new_state["info"]["step_number"])] = actions
        new_state["info"]["collision_history"].append(sum(collision_flags))
        new_state["info"]["agent_path_history"] = [new_state["info"]["agent_path_history"][i] + [new_positions[i]] for i in range(num_agents)]
        new_state["info"]["is_done"] = all(new_state["info"]["current_agent_positions"][i] == new_state["info"]["goal_positions"][i] for i in range(num_agents)) or new_state["info"]["step_number"] >= new_state["info"]["max_steps"]

        observation = self._format_observation_message(new_state, actions, collision_flags)
        return [{"role":"user","content":observation}], new_state


    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        info = state.get("info")
        if not info:
            raise ValueError("State missing 'info' field")
        return info.get("is_done")
    
    @staticmethod
    def _format_observation_message(state: State, actions: List[int], collisions: List[bool]) -> str:
        info = state.get("info")
        if not info:
            raise ValueError("State missing 'info' field")

        num_agents = len(info["current_agent_positions"])
        parts = [f"Step {info['step_number']}/{info['max_steps']} - Actions executed:"]
        action_names = {0:"LEFT",1:"UP",2:"STAY",3:"DOWN",4:"RIGHT"}
        for i in range(num_agents):
            act = actions[i] if i < len(actions) else 2
            col = collisions[i] if i < len(collisions) else False
            pos = info["current_agent_positions"][i]
            at_goal = pos == info["goal_positions"][i]
            name = action_names.get(act, f"UNKNOWN({act})")
            if col:
                desc = "❌ COLLISION - stayed"
            elif at_goal:
                desc = "✅ AT GOAL"
            else:
                desc = "✅ Moved"
            parts.append(f" Agent {i}: Action={name} result={desc} at position {pos}")
        parts.append("")
        parts.append(f"Grid size: {info['width']}x{info['height']}.")
        parts.append(f"Obstacles at: {info['obstacles']}.")
        parts.append(f"Agent goal positions: {info['goal_positions']}.")
        parts.append(f"Agents at goals: {sum(1 for i in range(num_agents) if info['current_agent_positions'][i] == info['goal_positions'][i])}/{num_agents}.")
        parts.append("Don't move agents who are at their goals, use 2 (STAY) for them.")
        parts.append("")

        parts.append(f"Agents and their start/goal positions:")
        for i in range(num_agents):
            parts.append(f"Agent {i}: From {info['current_agent_positions'][i]} to {info['goal_positions'][i]}")

        parts.append("")
        parts.append("Reminder of action format:")
        parts.extend([
            "Example of movement:",
            " Starting at position (2,3),",
            " - action 0 (LEFT) moves to (1,3)",
            " - action 1 (UP) moves to (2,2)",
            " - action 2 (STAY) stays at (2,3)",
            " - action 3 (DOWN) moves to (2,4)",
            " - action 4 (RIGHT) moves to (3,3)",
            "The x coordinate changes with LEFT/RIGHT, and the y coordinate changes with UP/DOWN."
        ])

        parts.append("")
        parts.append("STRICT OUTPUT FORMAT INSTRUCTIONS:")
        parts.append(f"Return EXACTLY {num_agents} integers in <actions>...</actions> tag.")
        parts.append("Use <Think>...</Think> tags for any internal reasoning, but not for actions.")
        parts.append("No other text outside <actions>...</actions> and optional <Think>...</Think> tags.")
        parts.append("No JSON, YAML, or other markup - just plain text.")
        parts.append("ALWAYS INCLUDE <actions>...</actions> tags.")
        parts.append("Only these integers allowed (no extra text): 0,1,2,3,4.")
        parts.append("If too few integers, pad with 2 (STAY). Too many, truncate.")
        
        if all(info["current_agent_positions"][i] == info["goal_positions"][i] for i in range(num_agents)):
            parts.append("🎉 All agents reached goals! Episode complete.")
        elif info["step_number"] >= info["max_steps"]:
            parts.append("⏰ Max steps reached. Episode over.")
        else:
            parts.append("Provide your next move actions inside <actions>…</actions> tags.")
        return "\n".join(parts)

    @staticmethod
    def _build_scenario_from_game(state: State) -> ScenarioConfig:

        info = state.get("info")
        if not info:
            raise ValueError("State missing 'info' field")

        sc = ScenarioConfig.from_dict(info["original_scenario"])

        current_agent_positions = [tuple(p) for p in info["current_agent_positions"]]

        sc.agent_positions = current_agent_positions

        return sc

    @staticmethod
    def extract_actions(text: str, num_agents: Optional[int] = None) -> Optional[List[int]]:
        if not text:
            print("No text to extract actions from.")
            return []
        m = re.search(r"<actions>(.*?)</actions>", text, re.DOTALL | re.IGNORECASE)
        if not m:
            print("No <actions>...</actions> tag found.")
            return []
        body = m.group(1).strip()
        parts = [p.strip() for p in re.split(r"[, \n\t]+", body) if p]
        try:
            ints = [int(p) for p in parts]
        except ValueError:
            print("Failed to convert action parts to integers.")
            return []
        if any(a not in Agent.actions for a in ints):
            print(f"Invalid actions found: {ints}")
            return []
        return ints

    def _make_rubric(self) -> vf.Rubric:
        def success_reward(state, **kwargs):
            info = state.get("info")
            if not info:
                raise ValueError("State missing 'info' field")
            num_agents = len(info["current_agent_positions"])
            agents_at_goal = sum(
                1 for i in range(num_agents) 
                if info["current_agent_positions"][i] == info["goal_positions"][i]
            )
            return agents_at_goal / num_agents if num_agents > 0 else 0.0

        def progress_reward(state, **kwargs):
            info = state.get("info")
            if not info:
                raise ValueError("State missing 'info' field")

            num_agents = len(info["current_agent_positions"])
            if num_agents == 0:
                return 0.0

            initial_positions = [path[0] for path in info["agent_path_history"] if path]

            total_progress = 0.0
            for i in range(num_agents):

                init_pos = initial_positions[i]
                curr_pos = info["current_agent_positions"][i]
                goal = info["goal_positions"][i]

                initial_dist = abs(init_pos[0] - goal[0]) + abs(init_pos[1] - goal[1])
                current_dist = abs(curr_pos[0] - goal[0]) + abs(curr_pos[1] - goal[1])

                progress = (initial_dist - current_dist) / initial_dist if initial_dist > 0 else 0.0
                total_progress += progress

            return total_progress / num_agents

        def collision_penalty(state, **kwargs):
            info = state.get("info")
            if not info:
                raise ValueError("State missing 'info' field")

            total_collisions = sum(info["collision_history"])
            if total_collisions == 0:
                return 0.0

            num_agents = len(info["current_agent_positions"])
            max_steps = info["max_steps"]
            max_possible = num_agents * max_steps
            return -(total_collisions / max_possible) * 0.3

        return vf.Rubric(
            funcs=[success_reward, progress_reward, collision_penalty],
            weights=[0.8, 0.2, 0.2]
        )

        
def create_train_scenarios(num_scenarios: int, grid_sizes: List[Tuple[int, int]],
                           agent_counts: List[int], obstacle_densities: List[float],
                           max_tries: int, max_steps: int, seed: int, scenario_verbose: bool) -> List[ScenarioConfig]:
    generator = ScenarioGenerator(seed=seed)
    scenarios = generator.generate_dataset(
        num_scenarios=num_scenarios,
        grid_sizes=grid_sizes,
        agent_counts=agent_counts,
        obstacle_densities=obstacle_densities,
        max_tries=max_tries,
        max_steps=max_steps,
        scenario_verbose=scenario_verbose
    )
    return scenarios

def load_eval_scenarios(filename: str) -> List[ScenarioConfig]:
    generator = ScenarioGenerator()
    return generator.load_dataset(filename)

def _format_initial_message(scenario: ScenarioConfig) -> str:
    num_agents = len(scenario.agent_positions) if hasattr(scenario, "agent_positions") else 0
    parts = [
        "You are a multi-agent planning coordinator. Your goal is to compute collision-free moves to get all agents to their goals on the grid.",
        "",
        "Uses top-left origin coordinate system where (0,0) is top-left, x increases rightward, and y increases downward.",
        "",
        f"Grid size: {scenario.width}x{scenario.height}.",
        f"Agents: {num_agents}.",
        f"Agent start positions: {scenario.agent_positions}",
        f"Agent goal positions: {scenario.goal_positions}",
        f"Obstacles at: {scenario.obstacles}.",
        f"Maximum steps per episode: {scenario.max_steps}.",
        "",
        "Provide actions for all agents as comma separated integers inside <actions>...</actions> tags.",
        "Actions: 0=LEFT,1=UP,2=STAY,3=DOWN,4=RIGHT.",
        "",
        "Example of movement:",
        " Starting at position (2,3),",
        " - action 0 (LEFT) moves to (1,3)",
        " - action 1 (UP) moves to (2,2)",
        " - action 2 (STAY) stays at (2,3)",
        " - action 3 (DOWN) moves to (2,4)",
        " - action 4 (RIGHT) moves to (3,3)",
        "The x coordinate changes with LEFT/RIGHT, and the y coordinate changes with UP/DOWN.",
    ]
    parts.append("")
    parts.append("Agents and their start/goal positions:")
    for i in range(num_agents):
        parts.append(f"Agent {i}: From {scenario.agent_positions[i]} to {scenario.goal_positions[i]}")
    parts.append("")
    parts.append("Think carefully about collisions and avoid them. Think like a simple pathfinding algorithm like A*. And think of the best move for each agent to get closer to its goal.")
    parts.append("STRICT OUTPUT FORMAT INSTRUCTIONS:")
    parts.append(f"Return EXACTLY {num_agents} integers in <actions>...</actions> tag.")
    parts.append("Use <Think>...</Think> tags for any internal reasoning, but not for actions.")
    parts.append("No other text outside <actions>...</actions> and optional <Think>...</Think> tags.")
    parts.append("No JSON, YAML, or other markup - just plain text.")
    parts.append("ALWAYS INCLUDE <actions>...</actions> tags.")
    parts.append("Only these integers allowed (no extra text): 0,1,2,3,4.")
    parts.append("If too few integers, pad with 2 (STAY). Too many, truncate.")
    parts.append("Start planning your first move now.")
    return "\n".join(parts)

from datasets import Dataset

def load_dataset(num_train: int, train_grid_sizes: List[Tuple[int, int]],
                 train_agent_counts: List[int], train_obstacle_densities: List[float],
                 train_max_tries: int, train_max_steps: int, train_seed: int,
                 eval_filename: Optional[str] = None, scenario_verbose: bool = False) -> Tuple[Dataset, Dataset]:

    train_scenarios = create_train_scenarios(
        num_scenarios=num_train,
        grid_sizes=train_grid_sizes,
        agent_counts=train_agent_counts,
        obstacle_densities=train_obstacle_densities,
        max_tries=train_max_tries,
        max_steps=train_max_steps,
        seed=train_seed,
        scenario_verbose=scenario_verbose
    )

    train_rows = []
    for scenario in train_scenarios:
        formatted_str = _format_initial_message(scenario)
        original_scenario = scenario.to_dict()
        train_rows.append({
        "question": formatted_str,
        "info": {
            "original_scenario": original_scenario,
            "goal_positions": scenario.goal_positions,
            "current_agent_positions": scenario.agent_positions,
            "obstacles": scenario.obstacles,
            "width": scenario.width,
            "height": scenario.height,
            "max_steps": scenario.max_steps,
            "step_number": 0,
            "collision_history": [],
            "action_history": {},
            # Initialize each agent's path history to start with their starting position
            "agent_path_history": [[pos] for pos in scenario.agent_positions],
            "is_done": False
        },
        "task": "multi-agent-planning"
    })

    train_dataset = Dataset.from_list(train_rows)

    if eval_filename is not None:
        eval_scenarios = load_eval_scenarios(eval_filename)
        eval_rows = []
        for scenario in eval_scenarios:
            formatted_str = _format_initial_message(scenario)
            original_scenario = scenario.to_dict()
            eval_rows.append({
                "question": formatted_str,
                "info": {
                    "original_scenario": original_scenario,
                    "goal_positions": scenario.goal_positions,
                    "current_agent_positions": scenario.agent_positions,
                    "obstacles": scenario.obstacles,
                    "width": scenario.width,
                    "height": scenario.height,
                    "max_steps": scenario.max_steps,
                    "step_number": 0,
                    "collision_history": [],
                    "action_history": {},
                    # Initialize each agent's path history to start with their starting position
                    "agent_path_history": [[pos] for pos in scenario.agent_positions],
                    "is_done": False
                },
                "task": "multi-agent-planning"
            })
        eval_dataset = Dataset.from_list(eval_rows)
    else:
        eval_dataset = Dataset.from_list([])

    return train_dataset, eval_dataset

def load_environment(**kwargs) -> vf.Environment:
    # Extract parameters or use defaults (and remove them from kwargs so we don't pass duplicates)
    num_train = kwargs.pop("num_scenarios", 128)
    train_grid_sizes = kwargs.pop("grid_sizes", [(5, 5), (6, 6)])
    train_agent_counts = kwargs.pop("agent_counts", [2, 3, 4])
    train_obstacle_densities = kwargs.pop("obstacle_densities", [0.1, 0.2, 0.3])
    train_max_tries = kwargs.pop("max_tries", 200)
    train_max_steps = kwargs.pop("max_steps", 32)
    train_seed = kwargs.pop("seed", 42)
    scenario_verbose = kwargs.pop("scenario_verbose", False)
    collision_verbose = kwargs.pop("collision_verbose", False)
    eval_filename = kwargs.pop("eval_filename", None)

    train_data, eval_data = load_dataset(
        num_train=num_train,
        train_grid_sizes=train_grid_sizes,
        train_agent_counts=train_agent_counts,
        train_obstacle_densities=train_obstacle_densities,
        train_max_tries=train_max_tries,
        train_max_steps=train_max_steps,
        train_seed=train_seed,
        eval_filename=eval_filename,
        scenario_verbose=scenario_verbose
    )

    dataset = train_data
    eval_dataset = eval_data

    return MappEnv(
        system_prompt="You are a multi-agent planning coordinator. Your goal is to compute collision-free moves to get all agents to their goals on the grid.",
        dataset=dataset,
        eval_dataset=eval_dataset,
        collision_verbose=collision_verbose,
        **kwargs
    )
