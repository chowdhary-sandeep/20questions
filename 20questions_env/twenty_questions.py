"""
twenty-questions — Prime Intellect Environments Hub
A multi-turn 20 Questions game where an LLM agent identifies a secret word
by asking yes/no questions answered by an LLM oracle.

Reward: sparse, episode-level only.
  - Correct guess: 1.0 + 0.5 * (questions_remaining / 20)  → range [1.0, 1.5]
  - Wrong / timeout: 0.0
"""

import json
import os
import random
import re
from pathlib import Path

import verifiers as vf
from datasets import Dataset
from openai import AsyncOpenAI

# ── bundled pool ──────────────────────────────────────────────────────────────
_POOL_PATH = Path(__file__).parent / "pool" / "pool_enriched.json"

# ── prompts ───────────────────────────────────────────────────────────────────
_ORACLE_SYSTEM = """You are the oracle in a game of 20 Questions.
The secret concept is: **{word}**

IMPORTANT — your entire response must be a SINGLE WORD, nothing else.
Allowed responses: Yes | No | Sometimes | Unclear
Do NOT write explanations, punctuation, tags, or any other text.
Do NOT reveal the concept name."""

DEFAULT_SYSTEM_PROMPT = """You are playing 20 Questions. A secret concept has been chosen. Identify it.

STRICT FORMAT — every reply must contain EXACTLY ONE of these two tags, nothing else:
  Ask a question : <question>Is it a living thing?</question>
  Make a guess   : <guess>elephant</guess>

Strategy:
- Ask binary yes/no questions that cut the remaining possibilities in half.
- Narrow category first (living/non-living, animal/object, etc.), then specifics.
- Guess only when confident. Earlier correct guesses score higher.
- A wrong guess counts as one of your 20 turns — it will be answered "No, that's not it."
- Do NOT output any text outside the tag."""

_INITIAL_USER_MSG = "I'm thinking of something. Ask me yes/no questions to identify it!"

MAX_QUESTIONS = 20

# ── helpers ───────────────────────────────────────────────────────────────────
_Q_RE = re.compile(r"<question>(.*?)</question>", re.DOTALL | re.IGNORECASE)
_G_RE = re.compile(r"<guess>(.*?)</guess>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_PLACEHOLDERS = re.compile(
    r"[.\s]*|piano|word|your answer here|your guess|your_guess", re.IGNORECASE
)


def _parse_action(text: str) -> tuple[str, str]:
    """Returns ('question'|'guess'|'invalid', content)."""
    text = _THINK_RE.sub("", text).strip()
    gs = _G_RE.findall(text)
    if gs:
        return ("guess", gs[-1].strip())
    qs = _Q_RE.findall(text)
    if qs:
        return ("question", qs[-1].strip())
    return ("invalid", "")


def _is_correct(guess: str, target: str) -> bool:
    """Accept exact match, WordNet synonyms, hyponyms, or compound suffix."""
    g, t = guess.lower().strip(), target.lower().strip()
    if g == t:
        return True
    try:
        from nltk.corpus import wordnet as wn
        target_ss = set(wn.synsets(t, pos="n"))
        guess_ss = set(wn.synsets(g, pos="n"))
        if not target_ss or not guess_ss:
            pass
        elif target_ss & guess_ss:
            return True
        else:
            for gs in guess_ss:
                for path in gs.hypernym_paths():
                    if target_ss & set(path):
                        return True
            for ts in target_ss:
                if guess_ss & set(ts.hypernyms()):
                    return True
    except Exception:
        pass
    if len(t) >= 4 and g.endswith(t):
        return True
    return False


# ── reward function ───────────────────────────────────────────────────────────
def twenty_questions_reward(completion, answer, **kwargs) -> float:
    """
    Sparse episode-level reward.
    Walks the full completion (all messages after the initial prompt) and
    returns 1.0 + efficiency bonus if a correct guess was made, else 0.0.
    """
    word = answer  # state["answer"] = secret word string
    n_q = 0
    for m in completion:
        if m["role"] != "assistant":
            continue
        atype, atext = _parse_action(m["content"])
        if atype in ("question", "guess") and not re.fullmatch(_PLACEHOLDERS, atext):
            n_q += 1
        if atype == "guess" and _is_correct(atext, word):
            efficiency = (MAX_QUESTIONS - n_q) / (MAX_QUESTIONS - 1)
            return round(1.0 + 0.5 * efficiency, 4)
    return 0.0


# ── dataset builder ───────────────────────────────────────────────────────────
def _build_datasets(
    pool: list[dict],
    num_train: int,
    num_eval: int,
    seed: int,
) -> tuple[Dataset, Dataset]:
    rng = random.Random(seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)

    # hold out eval set, then take up to num_train for training
    n_eval = min(num_eval, max(0, len(shuffled) - 1))
    eval_rows = shuffled[:n_eval]
    train_rows = shuffled[n_eval:][:num_train]

    def to_dataset(rows: list[dict]) -> Dataset:
        # Build initial user message per example; system prompt is prepended by
        # the framework via system_prompt= passed to Environment.__init__.
        prompts = [
            [{"role": "user", "content": _INITIAL_USER_MSG}]
            for _ in rows
        ]
        return Dataset.from_dict({
            "prompt": prompts,
            "answer": [r["word"] for r in rows],
            "difficulty": [r["difficulty"] for r in rows],
            "tier": [r["tier"] for r in rows],
        })

    return to_dataset(train_rows), to_dataset(eval_rows)


# ── env class ─────────────────────────────────────────────────────────────────
class TwentyQuestionsEnv(vf.MultiTurnEnv):
    """
    Multi-turn 20 Questions environment.

    The agent asks yes/no questions (or makes guesses). Each turn the oracle
    answers. Wrong guesses count as a turn and are answered "No". A correct
    guess ends the episode immediately with reward 1.0–1.5. Running out of
    questions gives reward 0.0.
    """

    def __init__(
        self,
        oracle_client: AsyncOpenAI,
        oracle_model: str,
        **kwargs,
    ):
        self._oracle_client = oracle_client
        self._oracle_model = oracle_model
        super().__init__(**kwargs)

    async def env_response(self, messages, state, **kwargs):
        """
        Called after each agent turn. Returns oracle answer or game-end message.
        Sets state["final_env_response"] on correct guess to end the episode.
        """
        word = state["answer"]
        last = next(
            (m["content"] for m in reversed(messages) if m["role"] == "assistant"), ""
        )
        action_type, action_text = _parse_action(last)

        # placeholder / empty guess → invalid feedback
        if action_type == "guess" and re.fullmatch(_PLACEHOLDERS, action_text):
            response = "That wasn't a real guess. Use <guess>word</guess>."

        # correct guess → signal episode end
        elif action_type == "guess" and _is_correct(action_text, word):
            msg = [{"role": "user", "content": "[CORRECT] Yes, that's it!"}]
            state["final_env_response"] = msg
            return msg

        # wrong guess → counts as a turn, oracle answers "No"
        elif action_type == "guess":
            response = "No, that's not it."

        # question → call oracle
        elif action_type == "question":
            resp = await self._oracle_client.chat.completions.create(
                model=self._oracle_model,
                messages=[
                    {"role": "system", "content": _ORACLE_SYSTEM.format(word=word)},
                    {"role": "user", "content": action_text},
                ],
                temperature=0.0,
                max_tokens=64,
            )
            content = (resp.choices[0].message.content or "").strip()
            content = _THINK_RE.sub("", content).strip()
            m = re.search(r"\b(yes|no|sometimes|unclear)\b", content, re.IGNORECASE)
            response = m.group(1).capitalize() if m else "Unclear"

        # invalid format
        else:
            response = "Invalid format. Use <question>...</question> or <guess>...</guess>."

        return [{"role": "user", "content": response}]


# ── loader ────────────────────────────────────────────────────────────────────
def load_environment(
    tier: int = 1,
    oracle_model: str = "gpt-4.1-mini",
    oracle_base_url: str = "https://api.openai.com/v1",
    oracle_api_key_var: str = "OPENAI_API_KEY",
    num_train_examples: int = 2000,
    num_eval_examples: int = 50,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    seed: int = 0,
    **kwargs,
) -> TwentyQuestionsEnv:
    """
    Load the 20 Questions environment.

    Args:
        tier: Word difficulty tier (percentile-based on combined difficulty score).
               1 = easiest 1%  (~81 words: common + concrete, e.g. baby, ball, fish)
               2 = 1st–5th pct (~326 words)
               3 = 5th–10th pct (~408 words)
               4 = 10th–20th pct (~816 words)
               5 = hardest, > 20th pct (~6,524 words: rare and/or abstract)
        oracle_model: Model used to answer yes/no questions.
        oracle_base_url: API base URL for the oracle model.
        oracle_api_key_var: Environment variable name holding the oracle API key.
        num_train_examples: Words sampled for training.
        num_eval_examples: Words sampled for evaluation.
        system_prompt: System prompt for the player agent.
        seed: Random seed for dataset sampling.
    """
    # load & filter pool
    with open(_POOL_PATH) as f:
        pool_all = json.load(f)

    if tier not in range(1, 6):
        raise ValueError(f"tier must be 1–5, got {tier}")
    pool = [e for e in pool_all if e.get("tier") == tier]

    if not pool:
        raise ValueError(f"No words found for tier={tier}")

    # build datasets
    train_ds, eval_ds = _build_datasets(pool, num_train_examples, num_eval_examples, seed)

    # oracle client (async required for env_response)
    api_key = os.environ.get(oracle_api_key_var, "")
    oracle_client = AsyncOpenAI(api_key=api_key, base_url=oracle_base_url)

    parser = vf.XMLParser(fields=["question", "guess"], answer_field="guess")

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(twenty_questions_reward)

    env = TwentyQuestionsEnv(
        oracle_client=oracle_client,
        oracle_model=oracle_model,
        dataset=train_ds,
        eval_dataset=eval_ds,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        max_turns=MAX_QUESTIONS + 4,
        **kwargs,
    )
    return env
