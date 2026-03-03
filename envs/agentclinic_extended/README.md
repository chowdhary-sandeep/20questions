# agentclinic-extended

### Overview
- **Environment ID**: `agentclinic-extended`
- **Short description**: Extended AgentClinic environment with enhanced multiturn capabilities for MEDQA cases
- **Tags**: medical, multiturn, evaluation, verifiable-reward

### Datasets
- **Primary dataset(s)**: agentclinic_medqa_extended.jsonl (214 medical cases)
- **Source links**: AgentClinic project
- **Split sizes**: 214 cases total

### Task
- **Type**: single-turn (simplified for initial implementation)
- **Parser**: Boxed answer extraction
- **Rubric overview**: Accuracy-based evaluation with exact and fuzzy matching

### Quickstart
Run an evaluation with default settings:

```bash
vf-eval agentclinic-extended
```

Configure model and sampling:

```bash
vf-eval agentclinic-extended -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_path` | str | `None` | Path to the JSONL dataset file |
| `use_think` | bool | `False` | Whether to use think mode (step-by-step reasoning) |
| `max_turns` | int | `10` | Maximum number of turns per conversation |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `accuracy` | Exact match on target diagnosis (0.0 or 1.0) |

