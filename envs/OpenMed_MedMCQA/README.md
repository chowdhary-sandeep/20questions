# OpenMed_MedMCQA

### Overview
- **Environment ID**: `OpenMed_MedMCQA`
- **Short description**: Single-turn medical multiple-choice QA on MedMCQA with chain-of-thought and a final decision as a boxed letter `\\boxed{A|B|C|D}`.
- **Tags**: openmed, medmcqa, multiple-choice, single-turn, think, boxed-letter, train, eval

### Dataset
- **Source**: `medmcqa` (HF datasets)
- **Splits**: Uses provided `train`, `validation`, and `test` if present; otherwise creates a train/eval holdout from `train` via `train_test_split` (seed=42).
- **Fields used**:
  - `question`: the stem.
  - `opa`, `opb`, `opc`, `opd`: four option strings, mapped to A–D.
  - `cop`: correct option (typically `A|B|C|D`; numeric or text match also supported).
  - `exp`, `choice_type`, `subject_name`, `topic_name`: kept as-is but not required by the env.

### Prompting & Schema
- **System message**: Instructs to reason inside `<think>...</think>` and put the final choice letter in `\\boxed{...}` using exactly one token from `{A,B,C,D}`.
- **User message**: Built with the provided `doc_to_text`-style template: `Question: ...`, `Choices:` with `A. ...` to `D. ...`, ending with `Answer:`.
- **Example schema per example**:
  - `prompt`: list of messages `[{"role":"system",...}, {"role":"user",...}]`
  - `options`: list of 4 option strings
  - `answer_letter`: one of `A|B|C|D`
  - `answer_idx`: integer index (0–3)
  - `answer`: letter (e.g., `"C"`)

### Parser & Rewards
- **Parser**: `ThinkParser` with `extract_boxed_answer` to read the final letter from `\\boxed{...}`.
- **Rewards**:
  - `correct_letter_reward_func` (weight 1.0): 1.0 if parsed letter equals `answer_letter` (numeric `0–3` also accepted and mapped), else 0.0.
  - `parser.get_format_reward_func()` (weight 0.0): optional format adherence (not counted).

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `-1` | Limit training set size (`-1` for all) |
| `num_eval_examples` | int | `-1` | Limit eval set size (`-1` for all) |

### Quickstart

Evaluate with defaults (uses the env’s internal dataset handling):

```bash
uv run vf-eval OpenMed_MedMCQA \
  -a '{"num_train_examples":-1, "num_eval_examples":-1}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- Reports (if produced) will be placed under `./environments/OpenMed_MedMCQA/reports/`.
 - Choices are displayed in a deterministic randomized label order per example (seeded by `id`); the underlying option mapping (A→opa, B→opb, …) and targets remain unchanged.

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval OpenMed_MedMCQA -a '{"key": "value"}'</code> to generate one.</p>
<!-- vf:end:reports -->
