"""
twenty-questions — Prime Intellect Environments Hub  (v0.1 MVP)
"""

import json, math, re
from pathlib import Path

_POOL_PATH = Path(__file__).parent / "pool" / "pool.json"
MAX_QUESTIONS = 20


def load_pool(path: Path = _POOL_PATH) -> list:
    with open(path) as f:
        return json.load(f)


_Q_RE = re.compile(r'<question>(.*?)</question>', re.DOTALL | re.IGNORECASE)
_G_RE = re.compile(r'<guess>(.*?)</guess>',       re.DOTALL | re.IGNORECASE)


def parse_action(text: str) -> tuple:
    gs = _G_RE.findall(text)
    if gs:
        return ("guess", gs[-1].strip())
    qs = _Q_RE.findall(text)
    if qs:
        return ("question", qs[-1].strip())
    return ("invalid", "")


def is_correct_guess(guess: str, target: str) -> bool:
    """
    Accept:
      1. Exact match
      2. Synonym — guess and target share a WordNet synset  (car/automobile)
      3. Guess is more specific — target appears in guess's hypernym path
                                   (smartphone → cellular_phone → phone ✓)
      4. Guess is one level more general — guess is a direct hypernym of target
                                   (canine → dog ✓)
    """
    g = guess.lower().strip()
    t = target.lower().strip()
    if g == t:
        return True
    try:
        from nltk.corpus import wordnet as wn
        target_synsets = set(wn.synsets(t, pos='n'))
        guess_synsets  = set(wn.synsets(g, pos='n'))
        if not target_synsets or not guess_synsets:
            pass  # fall through to compound suffix check
        # 1. synonym
        if target_synsets & guess_synsets:
            return True
        # 2. guess is more specific (walk full hypernym path up from guess)
        for gs in guess_synsets:
            for path in gs.hypernym_paths():
                if target_synsets & set(path):
                    return True
        # 3. guess is one level more general (direct hypernym of target)
        for ts in target_synsets:
            if guess_synsets & set(ts.hypernyms()):
                return True
    except Exception:
        pass
    # 4. compound-word fallback: target is the root of a compound guess
    #    (smartphone → phone, sunflower → flower, bookcase → case)
    #    guard: target must be ≥4 chars to avoid spurious 3-letter suffix matches
    if len(t) >= 4 and g.endswith(t):
        return True
    return False


def reward(correct: bool, n_questions: int,
           max_questions: int = MAX_QUESTIONS) -> float:
    if not correct:
        return 0.0
    efficiency = (max_questions - n_questions) / max_questions
    return round(1.0 + 0.5 * efficiency, 4)


def oracle_system_prompt(word: str) -> str:
    return f"""You are the oracle in a game of 20 Questions.
The secret concept is: **{word}**

IMPORTANT — your entire response must be a SINGLE WORD, nothing else.
Allowed responses: Yes | No | Sometimes | Unclear
Do NOT write explanations, punctuation, tags, or any other text.
Do NOT reveal the concept name."""


PLAYER_SYSTEM_PROMPT = """You are playing 20 Questions. A secret concept has been chosen. Identify it.

STRICT FORMAT — every reply must contain EXACTLY ONE of these two tags, nothing else:
  Ask a question : <question>Is it a living thing?</question>
  Make a guess   : <guess>elephant</guess>

Strategy:
- Ask binary yes/no questions that cut the remaining possibilities in half.
- Narrow category first (living/non-living, animal/object, etc.), then specifics.
- Guess only when confident. Earlier correct guesses score higher.
- Do NOT output any text outside the tag."""


# verifiers shim (graceful if not installed)
try:
    import verifiers as vf

    class TwentyQuestionsEnv(vf.MultiTurnEnv):
        def __init__(self, pool_path=_POOL_PATH, **kwargs):
            self.pool = load_pool(pool_path)
            super().__init__(**kwargs)

        def load_dataset(self):
            return [{"word": e["word"], "difficulty": e["difficulty"]}
                    for e in self.pool]

        def get_system_prompt(self, example):
            return PLAYER_SYSTEM_PROMPT

        def rubric(self, trajectory, example):
            # both questions and guesses consume a turn
            n_q = sum(1 for m in trajectory
                      if m["role"] == "assistant"
                      and parse_action(m["content"])[0] in ("question", "guess"))
            last = next((m["content"] for m in reversed(trajectory)
                         if m["role"] == "assistant"), "")
            atype, atext = parse_action(last)
            correct = (atype == "guess" and is_correct_guess(atext, example["word"]))
            return reward(correct, n_q)

except ImportError:
    class TwentyQuestionsEnv:
        def __init__(self, pool_path=_POOL_PATH, **kwargs):
            self.pool = load_pool(pool_path)
