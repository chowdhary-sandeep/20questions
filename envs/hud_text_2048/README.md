# hud-text-2048

### Overview
- **Environment ID**: `hud-text-2048`
- **Short description**: Text-based 2048 game for agents to reach target tiles through strategic moves
- **Tags**: `game`, `strategy`, `text`, `2048`

### Datasets
- **Primary dataset(s)**: Built-in tasks with varying difficulty (64 to 512 tiles)
- **Source links**: Generated programmatically in environment loader
- **Split sizes**: 4 tasks (you can adjust these)

### Task
- **Type**: multi-turn, tool use
- **Parser**: ToolXMLParser with action validation
- **Rubric overview**: Task completion (80%), format compliance (15%), tool execution (5%)

### Quickstart

Make sure to export your `HUD_API_KEY` (you can generate one [here](https://app.hud.so).

Run an evaluation with default settings:

```bash
uv run vf-eval hud-text-2048
```

Configure model and sampling:

```bash
uv run vf-eval hud-text-2048 \
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

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of task completion, format compliance, tool execution) |
| `task_completion` | Logarithmic scale based on highest tile vs target (min(1.0, log(highest)/log(target))) |
| `tool_execution` | Ratio of successful tool calls |
| `format_compliance` | Score for correct XML formatting and action syntax |

### How It Works

This environment uses `hud-vf-gym`, an adapter that bridges HUD's MCP infrastructure with the Verifiers RL framework:

1. **Environment Loader** (`hud_text_2048.py`):
   - Defines 6 tasks with increasing difficulty (64 to 2048 tiles)
   - Creates a Verifiers-compatible dataset with prompts and MCP configurations
   - Points to the `hudevals/hud-text-2048` Docker image

2. **MCP Integration** (via `hud-vf-gym`):
   - Spawns Docker container running the text-2048 MCP server
   - Executes setup tools to initialize game state
   - Translates agent actions to MCP tool calls
   - Returns text-based observations after each move
   - Runs evaluation tools to compute rewards

3. **Action Mapping** (`config.yaml`):
   - Maps high-level agent actions (`left()`, `right()`) to MCP's `move` tool
   - Uses declarative YAML configuration

### Relevant Links

- [HUD Documentation](https://docs.hud.so)
- [Build Your Own Environment](https://docs.hud.so/build-environments)
- [HUD Python SDK](https://github.com/hud-evals/hud-python)
- [HUD VF Gym Adapter](https://github.com/hud-evals/hud-vf-gym)