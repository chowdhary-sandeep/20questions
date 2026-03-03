# Q Programming Language Environment

A verifiers environment for evaluating and training models on Q programming language problems with real code execution and test case validation.

### Overview
- **Environment ID**: `q-programming-language`
- **Short description**: Evaluate and train models on Q programming language problems with real code execution and test case validation
- **Tags**: programming, q-language, code-generation, test-execution, finance

### About Q Programming Language

Q is a vector programming language developed by Kx Systems, designed for high-performance data analysis and financial modeling. It features:

- **Concise, functional syntax** for rapid development
- **Built-in vector operations** for efficient data processing
- **Fast execution** on large datasets
- **Wide adoption** in quantitative finance and time-series analysis

Q is particularly valuable for financial modeling, real-time analytics, and time-series processing, but is underrepresented in general-purpose AI training data compared to mainstream languages like Python, Java, and C++.

More details can be found in the full technical report here: [Technical Report: Full-Stack Fine-Tuning for the Q Programming Language](https://arxiv.org/abs/2508.06813)

### Datasets
- **Primary dataset**: [SFT Python-Q Programming Problems](https://huggingface.co/datasets/morganstanley/sft-python-q-problems)
- **Source**: LeetCode-style algorithmic problems with solutions in both Python and Q
- **Split sizes**: 542 train / 136 test problems
- **Features**: 678 unique programming problems with comprehensive test cases

### Task
- **Type**: Single-turn code generation
- **Parser**: ThinkParser with custom Q code extraction (handles `<answer>` tags and ````q` code blocks)
- **Rubric overview**: Test case execution reward (percentage passed + perfect bonus)

### Quickstart

**Prerequisites:**
1. Install Q language interpreter from [code.kx.com/q/](https://code.kx.com/q/) (community license available)
2. Set environment variable: `export Q_EXECUTABLE_PATH="/path/to/your/q/executable"`

Run an evaluation with default settings:

```bash
uv run vf-eval q-programming-language
```

Configure model and sampling:

```bash
uv run vf-eval q-programming-language \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"num_train_examples": 50, "perfect_bonus": 1.0}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `use_think` | bool | `true` | Enable reasoning format with `<answer>` tags |
| `num_train_examples` | int | `-1` | Limit training examples (use -1 for all) |
| `num_eval_examples` | int | `-1` | Limit evaluation examples (use -1 for all) |
| `perfect_bonus` | float | `1.0` | Bonus reward for passing all test cases |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (test case pass rate + perfect bonus) |
| `test_case_pass_rate` | Percentage of test cases passed |
| `perfect_solutions` | Number of solutions passing all test cases |

### How It Works

1. **Problem Loading**: Loads Q programming problems from the Morgan Stanley dataset
2. **Code Generation**: Models generate Q code solutions (with optional reasoning)
3. **Code Extraction**: Parser extracts Q code from completions (handles reasoning format and code blocks)
4. **Test Execution**: Generated code is executed against test cases using the Q interpreter
5. **Reward Calculation**: Reward based on percentage of test cases passed + bonus for perfect solutions

### Research Context

This environment is based on the comprehensive Q language training work described in:
- **Paper**: [Technical Report: Full-Stack Fine-Tuning for the Q Programming Language](https://arxiv.org/abs/2508.06813)
- **Dataset**: [SFT Python-Q Programming Problems](https://huggingface.co/datasets/morganstanley/sft-python-q-problems)

The research demonstrates that specialized fine-tuning can significantly improve model performance on niche programming languages, with the best model achieving 59% pass@1 accuracy on Q problems, surpassing Claude Opus-4 by 29.5%.

### Setup Requirements

1. **Q Language Interpreter**: Download from [code.kx.com/q/](https://code.kx.com/q/) (free community license available)
2. **Environment Variable**: Set `Q_EXECUTABLE_PATH` to point to your Q executable
3. **Dataset Access**: The environment automatically downloads the dataset from Hugging Face

Example setup:
```bash
# Download and install Q from code.kx.com/q/
export Q_EXECUTABLE_PATH="/path/to/your/q/executable"

# Test the environment
uv run vf-eval q-programming-language -n 5
```

