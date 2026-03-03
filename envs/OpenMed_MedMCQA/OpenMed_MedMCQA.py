"""
OpenMed MedMCQA Environment

Medical multiple-choice QA environment using the lighteval/med_mcqa dataset.
Adapted from reference implementation patterns for improved reliability.
"""

import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer
from datasets import load_dataset

LETTER_INDICES = ["A", "B", "C", "D"]


def load_environment(num_train_examples: int = -1, num_eval_examples: int = -1):
    """
    Single-turn MedMCQA environment using the `lighteval/med_mcqa` dataset.

    - Uses the cleaner lighteval dataset with proper 1-indexed cop field
    - Formats each example as chat messages with step-by-step reasoning
    - Rewards exact match on the final letter parsed from \\boxed{...}
    """

    # Load dataset - use lighteval version which has cleaner data
    train_ds = load_dataset("lighteval/med_mcqa", split="train")
    val_ds = load_dataset("lighteval/med_mcqa", split="validation")

    if num_train_examples != -1:
        train_ds = train_ds.select(range(min(num_train_examples, len(train_ds))))
    if num_eval_examples != -1:
        val_ds = val_ds.select(range(min(num_eval_examples, len(val_ds))))

    system_prompt = """You are a medical QA assistant. Think step-by-step to arrive at the best answer.

Reason through the question carefully, then output your final answer inside \\boxed{} using exactly one letter: A, B, C, or D.

Example: \\boxed{B}"""

    def _map_example(example):
        """Map a raw example to the format expected by verifiers."""
        cop = example.get("cop", -1)

        # Validate cop field (1-indexed: 1, 2, 3, 4)
        if not isinstance(cop, int) or cop not in [1, 2, 3, 4]:
            return None

        question = (example.get("question") or "").strip()
        options = [
            (example.get("opa") or "").strip(),
            (example.get("opb") or "").strip(),
            (example.get("opc") or "").strip(),
            (example.get("opd") or "").strip(),
        ]

        if not question or not any(options):
            return None

        # Convert 1-indexed cop to 0-indexed answer_idx
        answer_idx = cop - 1
        answer_letter = LETTER_INDICES[answer_idx]

        # Build the prompt
        query = f"Give a letter answer among A, B, C or D.\nQuestion: {question}\n"
        for i, opt in enumerate(options):
            query += f"{LETTER_INDICES[i]}. {opt}\n"
        query += "Answer:"

        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": query},
        ]

        return {
            "prompt": messages,
            "answer": answer_letter,
            "answer_idx": answer_idx,
            "options": options,
        }

    # Map and filter datasets
    train_mapped = train_ds.map(
        _map_example,
        remove_columns=train_ds.column_names,
        load_from_cache_file=True,
    ).filter(lambda x: x is not None, load_from_cache_file=True)

    val_mapped = val_ds.map(
        _map_example,
        remove_columns=val_ds.column_names,
        load_from_cache_file=True,
    ).filter(lambda x: x is not None, load_from_cache_file=True)

    # Validate datasets are not empty
    if len(train_mapped) == 0:
        raise ValueError("Training dataset is empty after filtering.")
    if len(val_mapped) == 0:
        raise ValueError("Eval dataset is empty after filtering.")

    # Use MaybeThinkParser - works with Qwen3 which auto-processes <think> tags
    # Qwen3's chat template strips <think> tags, so we can't check for them directly
    parser = vf.MaybeThinkParser(extract_fn=extract_boxed_answer)

    def accuracy_reward(completion, answer, **kwargs):
        """
        Accuracy reward: 1.0 if parsed answer matches expected letter, 0.0 otherwise.
        """
        # Parse the completion to extract the answer
        parsed = (parser.parse_answer(completion) or "").strip().upper()

        # Normalize expected answer
        expected = answer.strip().upper() if isinstance(answer, str) else ""

        # Check if both are valid letters
        if parsed not in {"A", "B", "C", "D"}:
            return 0.0
        if expected not in {"A", "B", "C", "D"}:
            return 0.0

        return 1.0 if parsed == expected else 0.0

    def format_reward(completion, **kwargs):
        """
        Format reward: encourages proper \\boxed{} usage and sufficient reasoning.
        Qwen3 strips <think> tags via chat template, so we check for:
        1. Proper \\boxed{} format with valid letter
        2. Reasonable amount of reasoning text before the answer
        """
        import re

        # Handle completion as list of messages (extract assistant content)
        if isinstance(completion, list):
            # Get content from last assistant message
            text = ""
            for msg in reversed(completion):
                if msg.get("role") == "assistant":
                    text = msg.get("content", "")
                    break
            if not text:
                return 0.0
        else:
            text = completion

        # Check for proper \boxed{} format
        boxed_pattern = r"\\boxed\{([A-Da-d])\}"
        boxed_match = re.search(boxed_pattern, text)
        if not boxed_match:
            return 0.0

        # Check for reasonable reasoning (at least 50 chars before \boxed{})
        boxed_pos = text.find("\\boxed{")
        if boxed_pos < 50:  # Too short, probably no reasoning
            return 0.0

        # Get reasoning text before the boxed answer
        reasoning = text[:boxed_pos].strip()

        # Count sentences/clauses (rough heuristic for reasoning quality)
        sentences = len(re.findall(r'[.!?]', reasoning))
        if sentences >= 2:  # At least 2 sentences of reasoning
            return 1.0

        return 0.5  # Partial credit for some reasoning

    rubric = vf.Rubric(
        parser=parser,
        funcs=[accuracy_reward, format_reward],
        weights=[0.8, 0.2],  # 80% accuracy, 20% format
    )

    env = vf.SingleTurnEnv(
        dataset=train_mapped,
        eval_dataset=val_mapped,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

    return env
