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
import re
from pathlib import Path

import verifiers as vf
from openai import OpenAI

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


def _reward(correct: bool, n_questions: int) -> float:
    if not correct:
        return 0.0
    efficiency = (MAX_QUESTIONS - n_questions) / MAX_QUESTIONS
    return round(1.0 + 0.5 * efficiency, 4)


# ── oracle ────────────────────────────────────────────────────────────────────
def _call_oracle(client: OpenAI, model: str, word: str, question: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _ORACLE_SYSTEM.format(word=word)},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=64,
    )
    content = (resp.choices[0].message.content or "").strip()
    content = _THINK_RE.sub("", content).strip()
    m = re.search(r"\b(yes|no|sometimes|unclear)\b", content, re.IGNORECASE)
    return m.group(1).capitalize() if m else "Unclear"


# ── env class ─────────────────────────────────────────────────────────────────
class TwentyQuestionsEnv(vf.MultiTurnEnv):
    """
    Multi-turn 20 Questions environment.

    The agent asks yes/no questions (or makes guesses). Each turn the oracle
    answers. Wrong guesses count as a turn and are answered "No". A correct
    guess ends the episode with reward 1.0–1.5. Running out of questions gives
    reward 0.0.
    """

    def __init__(
        self,
        pool: list[dict],
        oracle_client: OpenAI,
        oracle_model: str,
        **kwargs,
    ):
        self._pool = pool
        self._oracle_client = oracle_client
        self._oracle_model = oracle_model
        super().__init__(**kwargs)

    # ── verifiers interface ────────────────────────────────────────────────────
    def load_dataset(self) -> list[dict]:
        return [{"word": e["word"], "difficulty": e["difficulty"]} for e in self._pool]

    def get_system_prompt(self, example: dict) -> str:
        return DEFAULT_SYSTEM_PROMPT

    def env_response(self, messages: list[dict], answer: dict, **kwargs) -> str:
        """
        Called after each agent turn. Returns oracle answer or game-end message.
        `answer` is the example dict {"word": ..., "difficulty": ...}.
        """
        word = answer["word"]
        last = next(
            (m["content"] for m in reversed(messages) if m["role"] == "assistant"), ""
        )
        action_type, action_text = _parse_action(last)

        # placeholder / empty guess → invalid feedback
        if action_type == "guess" and re.fullmatch(_PLACEHOLDERS, action_text):
            return "That wasn't a real guess. Use <guess>word</guess>."

        # correct guess → signal end
        if action_type == "guess" and _is_correct(action_text, word):
            return f"[CORRECT] Yes, that's it!"

        # wrong guess → counts as turn with "No" answer
        if action_type == "guess":
            return f"No, that's not it."

        # question → call oracle
        if action_type == "question":
            return _call_oracle(
                self._oracle_client, self._oracle_model, word, action_text
            )

        # invalid format
        return "Invalid format. Use <question>...</question> or <guess>...</guess>."

    def rubric(self, trajectory: list[dict], example: dict) -> float:
        word = example["word"]
        n_q = 0
        for m in trajectory:
            if m["role"] != "assistant":
                continue
            atype, atext = _parse_action(m["content"])
            if atype in ("question", "guess") and not re.fullmatch(
                _PLACEHOLDERS, atext
            ):
                n_q += 1
            if atype == "guess" and _is_correct(atext, word):
                return _reward(True, n_q)
        return 0.0


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
        tier: Word difficulty tier.
               1 = concrete objects (concreteness >= 4.0, ~4,200 words)
               2 = moderate (3.0–4.0, ~2,800 words)
               3 = borderline abstract (2.5–3.0, ~1,200 words)
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

    tier_ranges = {1: (4.0, 9.9), 2: (3.0, 4.0), 3: (2.5, 3.0)}
    lo, hi = tier_ranges.get(tier, (4.0, 9.9))
    pool = [e for e in pool_all if lo <= e.get("concreteness", 0.0) < hi]

    if not pool:
        raise ValueError(f"No words found for tier={tier}")

    # oracle client
    api_key = os.environ.get(oracle_api_key_var, "")
    oracle_client = OpenAI(api_key=api_key, base_url=oracle_base_url)

    parser = vf.XMLParser(fields=["question", "guess"], answer_field="guess")
    rubric = vf.Rubric(parser=parser)

    env = TwentyQuestionsEnv(
        pool=pool,
        oracle_client=oracle_client,
        oracle_model=oracle_model,
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        seed=seed,
        max_turns=MAX_QUESTIONS + 4,
        **kwargs,
    )
    return env
