import json
import os
import verifiers as vf
from datasets import Dataset
from hud_vf_gym import HUDGym


def load_environment(**kwargs) -> vf.Environment:
    '''Load browser-based 2048 environment.'''
    
    tasks = [
        {
            "prompt": "Play 2048 and reach the 64 tile",
            "target": 64,
            "setup_name": "game_2048_board",
            "setup_args": {"board_size": 4, "target_tile": 64},
            "eval_name": "game_2048_max_number",
            "eval_args": {"target": 64}
        },
        {
            "prompt": "Play 2048 and reach the 128 tile",
            "target": 128,
            "setup_name": "game_2048_board",
            "setup_args": {"board_size": 4, "target_tile": 128},
            "eval_name": "game_2048_max_number",
            "eval_args": {"target": 128}
        },
        {
            "prompt": "Play 2048 and reach the 256 tile",
            "target": 256,
            "setup_name": "game_2048_board",
            "setup_args": {"board_size": 4, "target_tile": 256},
            "eval_name": "game_2048_max_number",
            "eval_args": {"target": 256}
        },
    ]
    
    dataset_dict = {
        "question": [],  # Task prompts
        "task": [],      # Task identifiers  
        "answer": [],    # Expected answers (empty for games)
        "info": []       # MCP config and tools as JSON strings
    }
    
    for i, task in enumerate(tasks):
        dataset_dict["question"].append(task["prompt"])
        dataset_dict["task"].append(f"browser_2048_{task['target']}")
        dataset_dict["answer"].append("")  # No expected text answer for games
        
        # Build info dict with JSON strings
        info = {
            "mcp_config": json.dumps({
                "local": {
                    "command": "sh",
                    "args": ["-c", "docker run --rm --platform linux/amd64 -i hudevals/hud-browser:0.1.3 2>/dev/null"]
                }
            }),
            
            # Setup tool - launch the app
            "setup_tool": json.dumps(
                {"name": "launch_app", "arguments": {"app_name": "2048"}}
            ),
            
            # Evaluate tool
            "evaluate_tool": json.dumps({
                "name": "evaluate",
                "arguments": {
                    "name": task["eval_name"],
                    "arguments": task["eval_args"]
                }
            }),
            
            # Optional metadata
            "metadata": json.dumps({"target_tile": task["target"]})
        }
        dataset_dict["info"].append(info)
    
    dataset = Dataset.from_dict(dataset_dict)
    

    config_path = kwargs.pop("config_path", None)
    if not config_path:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.yaml")

    return HUDGym(dataset=dataset, config_path=config_path, **kwargs)