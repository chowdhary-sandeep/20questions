# hud-browser-2048

### Overview
- **Environment ID**: `hud-browser-2048`
- **Short description**: Browser-based 2048 game for evaulating agents using visual observations and keyboard actions
- **Tags**: `game`, `browser`, `multimodal`, `CUA`, `2048`

### Datasets
- **Primary dataset(s)**: Built-in tasks with varying difficulty (64 to 256 tiles)
- **Source links**: Generated programmatically in environment loader
- **Split sizes**: 3 tasks (expandable)

### Task
- **Type**: multi-turn, tool use
- **Parser**: ToolXMLParser with action validation
- **Rubric overview**: Task completion (80%), format compliance (10%), tool execution (10%)

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval hud-browser-2048
```

Configure model and sampling:

```bash
uv run vf-eval hud-browser-2048 \
  -m gpt-4.1-mini \
  -n 1 -r 3 \
  -a '{"max_turns": 150}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_turns` | int | `100` | Maximum moves allowed per game |
| `system_prompt` | str | (see config) | Override agent instructions |
| `config_path` | str | `./config.yaml` | Path to alternative config file |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of task completion, format compliance, tool execution) |
| `task_completion` | Logarithmic scale based on highest tile vs target (min(1.0, log(highest)/log(target))) |
| `tool_execution` | Ratio of successful tool calls |
| `format_compliance` | Score for correct XML formatting and action syntax |

### How It Works

This environment uses `hud-vf-gym` to connect the browser-based 2048 game with the Verifiers RL framework:

1. **Browser Control**:
   - Uses `hudevals/hud-browser` Docker image with Playwright
   - Takes screenshots to observe game state
   - Sends keyboard inputs (arrow keys) to play

2. **Action Mapping**:
   - Agent calls `screenshot()` to see the game
   - Agent calls `up()`, `down()`, `left()`, `right()` for moves
   - Maps to environment's `computer` tool with appropriate key presses

3. **Task Flow**:
   - Setup: Launches 2048 web app in browser
   - Play: Agent uses screenshots and arrow keys
   - Evaluate: Checks if target tile was reached

The browser environment demonstrates HUD MCP's ability to wrap web applications as RL environments, enabling agents to learn from visual observations and keyboard interactions.

### Relevant Links

- [HUD Documentation](https://docs.hud.so)
- [Build Your Own Environment](https://docs.hud.so/build-environments)
- [Train Agents with Verifiers](https://docs.hud.so/train-agents)
- [HUD Python SDK](https://github.com/hud-evals/hud-python)
- [HUD VF Gym Adapter](https://github.com/hud-evals/hud-vf-gym)