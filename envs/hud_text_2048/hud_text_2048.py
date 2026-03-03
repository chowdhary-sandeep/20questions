import json
import os
import verifiers as vf
from datasets import Dataset
from hud_vf_gym import HUDGym


def load_environment(**kwargs) -> vf.Environment:
    '''Load text-based 2048 environment.'''
    
    tasks = [
        {
            "prompt": "Play 2048 and reach the 64 tile",
            "target": 64,
            "eval_args": {"target": 64}
        },
        {
            "prompt": "Play 2048 and reach the 128 tile",
            "target": 128,
            "eval_args": {"target": 128}
        },
        {
            "prompt": "Play 2048 and reach the 256 tile",
            "target": 256,
            "eval_args": {"target": 256}
        },
        {
            "prompt": "Play 2048 and reach the 512 tile",
            "target": 512,
            "eval_args": {"target": 512}
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
        dataset_dict["task"].append(f"text_2048_{task['target']}")
        dataset_dict["answer"].append("")  # No expected text answer for games
        
        # Build info dict with JSON strings
        info = {
            "mcp_config": json.dumps({
                "local": {
                    "command": "sh",
                    "args": ["-c", "docker run --rm --platform linux/amd64 -i hudevals/hud-text-2048:0.1.3 2>/dev/null"]
                }
            }),
            
            # Setup tool - initialize the board
            "setup_tool": json.dumps({
                "name": "setup",
                "arguments": {
                    "name": "board",
                    "arguments": {"board_size": 4}
                }
            }),
            
            # Evaluate tool - check if target reached
            "evaluate_tool": json.dumps({
                "name": "evaluate",
                "arguments": {
                    "name": "max_number",
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