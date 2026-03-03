# agentclinic-nejm

### Overview
- **Environment ID**: `agentclinic-nejm`
- **Short description**: AgentClinic NEJM environment with image-based medical cases for specialized medical evaluation
- **Tags**: medical, multiturn, evaluation, verifiable-reward, nejm, imaging

### Datasets
- **Primary dataset(s)**: agentclinic_nejm_extended.jsonl (120 NEJM medical cases with images)
- **Source links**: AgentClinic project, NEJM Clinical Images
- **Split sizes**: 120 cases total

### Task
- **Type**: multiturn (conversational medical diagnosis)
- **Parser**: Boxed answer extraction
- **Rubric overview**: Accuracy-based evaluation with exact and fuzzy matching for image-based medical cases

### Quickstart
Run an evaluation with default settings:

```bash
vf-eval agentclinic-nejm
```

Configure model and sampling:

```bash
vf-eval agentclinic-nejm -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7
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

### Special Features
- **Image-based cases**: Supports medical image analysis
- **Specialty tagging**: Cases tagged by medical specialty
- **Multiturn interaction**: Agent can ask for additional information
- **Comprehensive prompts**: Includes patient info, physical exams, and test results


    """
    Example CLI:
        uv pip install -e .             

      uv run --active -m verifiers.scripts.eval \
        -m mistral-small-latest \
        -b https://api.mistral.ai/v1 \
        -k MISTRAL_API_KEY \
        agentclinic_nejm  -n 120  --max-concurrent 4  --rollouts-per-example 3 -T 0.0 -s
    """