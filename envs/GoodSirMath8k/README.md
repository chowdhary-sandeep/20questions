# GoodSirMath8k


### Overview
- **Environment ID**: `GoodSirMath8k`
- **Short description**: Just GSM8K with the added reward based on how shakespearean the model is.
- **Tags**: math, gsm8k, single-turn, think, old-english


### Datasets
- **Primary dataset(s)**: gsm8k train (train) and test (eval) via load_example_dataset
- **Source links**: Uses the example loader in verifiers.utils.data_utils
- **Split sizes**: Configurable via args; defaults to full train/test

### Task
- **Type**: single-turn
- **Parser**: `XMLParser` with the tags `ponder` and `andswaru`
- **Rubric overview**: Exact match on parsed boxed answer, format check and classical english style reward.

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval GoodSirMath8k
```

Configure model and sampling:

```bash
uv run vf-eval GoodSirMath8k   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7 
```


### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `-1` | Limit training set size (-1 for all) |
| `num_eval_examples` | int | `-1` | Limit eval set size (use -1 for all) |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `correct_answer` | 1.0 if parsed boxed answer equals target, else 0.0 |
| `format_reward` | Adherence to `<ponder>` and `<andswaru>` tags  |
| `old_speak_reward` | Adherence to old style english, the DistilBERT-base-uncased finetuned provided in  `notaphoenix/shakespeare_classifier_model` is used to evaluate that|

