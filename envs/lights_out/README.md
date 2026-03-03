# lights-out

### Overview
- **Environment ID**: `lights-out`
- **Short description**: Grid-based game where clicking toggles a light & its neighbors. Goal: all lights off.
- **Tags**: eval, train, game, multi-turn, grid

### Datasets
- **Primary dataset(s)**: `scandukuri/lights-out-3x3`, a synthetic dataset of 3 x 3 Lights Out grids made from sampling random initial states and deterministically computing the canonical solution with `numpy`.
- **Source links**: [Dataset](https://huggingface.co/datasets/scandukuri/lights-out-3x3), [proof of solvability](https://www.jstor.org/stable/2687202) for particular board dimensions
- **Split sizes**: train split = `270`, test split = `30`

### Task
- **Type**: multi-turn
- **Parser**: `XMLParser`
- **Rubric overview**: The reward incorporates whether the model actually turns all lights off, whether the canonical solution was matched, how efficiently the board was solved, and whether responses were formatted correctly (`solved_reward`, `minimal_solution_reward`, `efficiency_reward`, `format_reward_func`)

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval lights-out
```

Configure model and sampling:

```bash
uv run vf-eval lights-out   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"key": "value"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_turns` | int | `12` | Max moves the model may make before the game ends (ends early if solved). |
| `use_think` | bool | `False` | Switches to a prompt that asks the model to think inside `<think>…</think>` before outputting `<step>row,col</step>`. |
| `show_canonical` | bool | `False` | Includes the minimal-solution step count in the prompt (when available). |
| `dataset_spec` | str | `scandukuri/lights-out-3x3` | HF dataset to load. Must supply square `initial_state` boards (0/1) and optionally `minimal_solution_steps`; size need not be 3×3. |


### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of all criteria) |
| `format_reward_func` | Checks whether the model produced a valid `<step>...</step>` output, inherited from `verifiers.parsers.xml_parser.XMLParser` (format compliance) |
| `solved_reward` | 1.0 if the board is solved at the end of the rollout, else 0.0 |
| `minimal_solution_reward` | 1.0 if solved using exactly the minimal number of moves, else 0.0 |
| `efficiency_reward` | Higher if solved in fewer moves relative to the `max_turns` budget (computed as `max(0.0, 1.0 - (turns_taken / max_turns))`) |

