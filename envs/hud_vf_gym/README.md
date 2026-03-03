# hud-vf-gym

### Overview
- **Environment ID**: `hud-vf-gym`
- **Short description**: Generic adapter that bridges HUD's MCP infrastructure with the Verifiers framework
- **Tags**: `adapter`, `mcp`, `CUA`

### Datasets
- **Primary dataset(s)**: Any HuggingFace dataset following HUD's task format
- **Source links**: Configurable via `taskset` parameter (e.g., `hud-evals/2048-taskset`)
- **Split sizes**: Depends on the loaded dataset

### Task
- **Type**: Configurable (single-turn or multi-turn, tool use)
- **Parser**: ToolXMLParser with configurable thinking mode
- **Rubric overview**: Weighted combination of task completion (from MCP evaluation), tool execution success, and format compliance

### Quickstart
Run an evaluation with any HUD-compatible taskset:

```bash
vf-eval hud-vf-gym \
  --env-args '{"taskset": "your-org/your-taskset", "config_path": "./configs/your-env.yaml"}' \
  --model gpt-4.1-mini \
```

Train an agent with GRPO (for text-based environments only):

```python
import verifiers as vf

env = vf.load_environment(
    env_id="hud-vf-gym",
    taskset="your-org/your-taskset",
    config_path="./configs/your-env.yaml"
)

model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-3B-Instruct")
trainer = vf.GRPOTrainer(
    model=model,
    env=env,
    args=vf.grpo_defaults()
)
trainer.train()
```

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `taskset` | str | (required) | HuggingFace dataset identifier |
| `config_path` | str | (required) | Path to environment configuration YAML |
| `num_tasks` | int | None | Optional limit on number of tasks to load |
| `split` | str | `"train"` | Dataset split to use |
| `max_turns` | int | (from config) | Override maximum turns per rollout |
| `system_prompt` | str | (from config) | Override agent instructions |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Weighted combination of all rubric components |
| `task_completion` | Score from MCP evaluation tool (environment-specific) |
| `tool_execution` | Ratio of successful tool calls to total attempts |
| `format_compliance` | XML format correctness and action syntax validation |

### How It Works

HUD VF Gym is a generic adapter that enables any MCP-compatible environment to work with the Verifiers RL framework. It provides the bridge between HUD's infrastructure and Verifiers' training/evaluation capabilities.

#### Components

1. **Main Module** (`hud_vf_gym.py`):
   - `load_environment()`: Loads tasks from HuggingFace datasets and creates HUDGym instance
   - `HUDGym`: Extends Verifiers' `MultiTurnEnv` base class
   - Manages Docker container lifecycle via MCP
   - Handles multi-turn agent-environment interactions
   - Integrates with HUD's telemetry and job tracking

2. **MCP Integration** (`utils/mcp_utils.py`):
   - `execute_tool()`: Universal tool execution through MCP protocol
   - `create_action_args()`: Maps agent actions to MCP tool arguments
   - Supports both direct MCP calls and action mapping transformations

3. **Parsing System** (`utils/parsers.py`):
   - `ToolXMLParser`: Validates XML-wrapped tool calls
   - Extracts actions from `<tool>action(args)</tool>` format
   - Configurable thinking mode with `<think>` tags
   - Combines XML validation with action syntax checking

4. **Reward System** (`utils/rubrics.py`):
   - `HUDBaseRubric`: Configurable weighted reward function
   - Components: task completion, tool execution, format compliance
   - Task completion comes from MCP evaluation tool
   - Tool execution tracks success rate
   - Format compliance validates XML and action syntax

#### Configuration System

Environments are configured through YAML files that define:

```yaml
# Job tracking
job:
  name: "Environment Run"
  metadata: {...}

# Agent instructions
system_prompt: |
  Instructions and available tools...

# Parser settings
parser:
  use_thinking: true/false  # Enable <think> tags
  xml_weight: 0.6           # XML format importance
  action_weight: 0.4        # Action syntax importance

# Action mappings - the core of configuration
action_mappings:
  agent_action:             # What the agent calls
    _tool: "mcp_tool"      # Underlying MCP tool
    _parser:
      positional: ["arg1"] # Expected arguments
    param1:
      from_arg: "arg1"     # Map from agent arg
      transform: "..."     # Optional transform
    param2:
      static: "value"      # Static value

# Rubric weights
rubric:
  weights:
    task_completion: 0.8
    tool_execution: 0.1
    format_compliance: 0.1
```

#### Rollout Process

1. **Initialization**:
   - Create MCP client with Docker container
   - Execute setup tools to prepare environment
   - Append setup results to initial prompt

2. **Multi-turn Loop**:
   - Agent generates XML-wrapped tool call
   - Parser extracts and validates action
   - Action mappings transform to MCP tool call
   - MCP executes tool, returns results
   - Results sent back to agent
   - Continue until done or max turns

3. **Evaluation**:
   - Execute evaluation tools
   - Compute rewards based on rubric
   - Clean up MCP resources

#### Key Features

- **Config-Driven**: No code changes needed for new environments
- **Action Mapping**: Declarative transformation from agent to MCP tools
- **Multimodal Support**: Handles text and image observations
- **Job Tracking**: [Automatic HUD telemetry integration](https://app.hud.so)

### Creating Custom Environments

To use hud-vf-gym with your own environment:

1. **Create a Docker image** with a MCP server that implements your environment through HUD SDK
2. **Define tasks** as a HuggingFace dataset with HUD format
3. **Write a config YAML** with action mappings for your tools
4. **Load and run**:
   ```python
   env = vf.load_environment(
       env_id="hud-vf-gym",
       taskset="your-org/your-taskset",
       config_path="your-config.yaml"
   )
   ```

### Also See

- [HUD Documentation](https://docs.hud.so)
- [Verifiers Framework](https://github.com/willccbb/verifiers)
- [HUD Python SDK](https://github.com/hud-evals/hud-python)
- [Example Configs](https://github.com/hud-evals/hud-python/tree/main/rl/configs)