# twenty-questions v0.1 — Build & Test Plan

## Design

The oracle is an LLM — it already knows what every word means. No knowledge
graph needed. The pool is just a clean list of English nouns with a difficulty
proxy derived from word frequency.

**Difficulty proxy**: Zipf frequency (wordfreq). Common words (`dog` ~5.5) are
easy — players narrow them quickly with yes/no questions. Rare words (`kumquat`
~2.1) are hard. `difficulty = 7.0 - zipf_frequency`.

**Optional enrichment**: Brysbaert concreteness ratings (40K words, 1–5 scale).
Abstract words (`justice` ~1.5) are harder than concrete ones (`dog` ~5.0)
because yes/no questions partition them poorly. Adds a second difficulty axis.

---

## Project Layout

```
E:\prime-envs\
├── twenty_questions_v0.md        ← this file
│
└── twenty_questions\             ← env package (mirrors Hub layout)
    ├── pyproject.toml
    ├── README.md
    ├── twenty_questions.py       ← env entry point (verifiers-compatible)
    ├── pool\
    │   ├── build_pool.py         ← WordNet + wordfreq → pool.json  (run once)
    │   └── pool.json             ← generated artifact (~3-5K concepts)
    └── .prime\
        └── .env-metadata.json    ← fake Hub metadata for local testing

test_twenty_questions.py          ← standalone test harness (NOT inside env)
test_results\
    ├── results_{model}_{timestamp}.json
    └── plots_{model}_{timestamp}.png
```

---

## Step 1 — Dependencies

```bash
uv pip install verifiers openai nltk wordfreq matplotlib numpy
python -c "import nltk; nltk.download('wordnet')"
```

`verifiers` — Prime Intellect env framework
`openai` — LM Studio / MiniMax both expose an OpenAI-compatible API
`nltk` — WordNet noun corpus
`wordfreq` — Zipf frequency for ~200K English words (SUBTLEX + Wikipedia + Twitter)

---

## Step 2 — Build the Concept Pool (`pool/build_pool.py`)

Runs once, takes ~10 seconds. Outputs `pool.json`.

```python
# pool/build_pool.py
"""
WordNet nouns + wordfreq → pool.json

Output schema per concept:
{
  "word":         "dog",
  "frequency":    5.43,      # Zipf scale (0=rare, 7=very common)
  "difficulty":   1.57,      # 7.0 - frequency  (higher = harder)
  "concreteness": 4.89       # Brysbaert 1-5 rating (omitted if CSV not found)
}

Run:
  python pool/build_pool.py
  python pool/build_pool.py --min-freq 2.5 --out pool/pool.json
  python pool/build_pool.py --concreteness path/to/concreteness.csv
"""

import argparse, json, os
from pathlib import Path


def build_pool(min_freq: float = 2.0,
               concreteness_csv: str | None = None) -> list:
    from nltk.corpus import wordnet as wn
    from wordfreq import zipf_frequency

    print("Extracting WordNet nouns…")
    nouns: set[str] = set()
    for synset in wn.all_synsets('n'):
        for lemma in synset.lemmas():
            word = lemma.name().lower()
            if '_' not in word and word.isalpha() and len(word) > 2:
                nouns.add(word)
    print(f"  {len(nouns):,} unique single-word nouns in WordNet")

    # Brysbaert concreteness ratings (optional)
    concreteness: dict[str, float] = {}
    if concreteness_csv and Path(concreteness_csv).exists():
        import csv
        with open(concreteness_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                w = row.get('Word', '').lower().strip()
                try:
                    concreteness[w] = float(row.get('Conc.M', 0))
                except ValueError:
                    pass
        print(f"  {len(concreteness):,} concreteness ratings loaded")

    print(f"Scoring with wordfreq (min Zipf >= {min_freq})…")
    pool = []
    for word in nouns:
        freq = zipf_frequency(word, 'en')
        if freq < min_freq:
            continue
        entry: dict = {
            "word":       word,
            "frequency":  round(freq, 4),
            "difficulty": round(7.0 - freq, 4),
        }
        if word in concreteness:
            entry["concreteness"] = round(concreteness[word], 2)
        pool.append(entry)

    pool.sort(key=lambda x: x["difficulty"])
    print(f"  Pool size: {len(pool):,} concepts  "
          f"(difficulty range: {pool[0]['difficulty']:.2f} – {pool[-1]['difficulty']:.2f})")
    return pool


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",           default=str(Path(__file__).parent / "pool.json"))
    ap.add_argument("--min-freq",      type=float, default=2.0)
    ap.add_argument("--concreteness",  default=None,
                    help="Path to Brysbaert concreteness CSV (optional)")
    args = ap.parse_args()

    pool = build_pool(min_freq=args.min_freq,
                      concreteness_csv=args.concreteness)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(pool, f, indent=2)
    print(f"Saved {len(pool):,} concepts -> {args.out}")
```

**Expected runtime**: ~10 seconds (all local, no downloads after `nltk.download('wordnet')`).
**Expected pool size**: 3,000–5,000 concepts at `--min-freq 2.0`.

---

## Step 3 — Environment File (`twenty_questions.py`)

Verifiers-compatible env. Loads `pool/pool.json`. Oracle prompt is just the
concept name — no property lists needed.

```python
# twenty_questions.py
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
  Make a guess   : <guess>piano</guess>

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
            n_q = sum(1 for m in trajectory
                      if m["role"] == "assistant"
                      and parse_action(m["content"])[0] == "question")
            last = next((m["content"] for m in reversed(trajectory)
                         if m["role"] == "assistant"), "")
            atype, atext = parse_action(last)
            correct = (atype == "guess"
                       and atext.lower().strip() == example["word"].lower().strip())
            return reward(correct, n_q)

except ImportError:
    class TwentyQuestionsEnv:
        def __init__(self, pool_path=_POOL_PATH, **kwargs):
            self.pool = load_pool(pool_path)
```

---

## Step 4 — Fake Hub Metadata (`.prime/.env-metadata.json`)

```json
{
  "id": "local-twenty-questions",
  "name": "twenty-questions",
  "owner": "bnwboi",
  "version": "0.1.0",
  "source": "local"
}
```

---

## Step 5 — Test Harness (`test_twenty_questions.py`)

```python
# test_twenty_questions.py
"""
Twenty Questions — Two-model test harness
Oracle:  LLM (knows the concept, answers Yes/No)
Player:  LLM (clean context, no concept knowledge)

Usage:
    python test_twenty_questions.py                    # 10 concepts
    python test_twenty_questions.py --n-concepts 50
    python test_twenty_questions.py --verbose
"""

import argparse, json, math, os, random, re, sys, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "twenty_questions"))
from twenty_questions import (
    load_pool, parse_action, reward,
    oracle_system_prompt, PLAYER_SYSTEM_PROMPT, MAX_QUESTIONS,
)

from openai import OpenAI

# ── Model config — edit here to swap backend ──────────────────────────────────

# Local (LM Studio) — active
API_KEY      = "lm-studio"
BASE_URL     = "http://127.0.0.1:1234/v1"
ORACLE_MODEL = "qwen3.5-35b-a3b@q5_k_xl"
PLAYER_MODEL = "qwen3.5-35b-a3b@q5_k_xl"

# MiniMax — uncomment to use
# API_KEY      = os.environ.get("MINIMAX_API_KEY", "")
# BASE_URL     = "https://api.minimaxi.chat/v1"
# ORACLE_MODEL = "MiniMax-M2.1"
# PLAYER_MODEL = "MiniMax-M2.5"

def make_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Oracle needs thinking ON to reason correctly; player has it OFF for speed
_ORACLE_EXTRA = {}
_PLAYER_EXTRA = {"extra_body": {"enable_thinking": False}}


def call_oracle(oracle_client, messages: list, system: str) -> str:
    resp = oracle_client.chat.completions.create(
        model=ORACLE_MODEL,
        messages=[{"role": "system", "content": system}] + messages,
        temperature=0.0,
        max_tokens=1024,
        **_ORACLE_EXTRA,
    )
    content = resp.choices[0].message.content or ""
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    m = re.search(r'\b(yes|no|sometimes|unclear)\b', content, re.IGNORECASE)
    return m.group(1).capitalize() if m else "Unclear"


def call_player(player_client, messages: list) -> str:
    resp = player_client.chat.completions.create(
        model=PLAYER_MODEL,
        messages=[{"role": "system", "content": PLAYER_SYSTEM_PROMPT}] + messages,
        temperature=0.7,
        max_tokens=2000,
        **_PLAYER_EXTRA,
    )
    content = resp.choices[0].message.content or ""
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return content


_PLACEHOLDERS = re.compile(
    r'[.\s]*|piano|your answer here|your guess|your_guess', re.IGNORECASE)


def run_game(oracle_client, player_client,
             entry: dict, verbose: bool = False) -> dict:
    word       = entry["word"]
    oracle_sys = oracle_system_prompt(word)

    player_msgs = [{"role": "user", "content": "The game begins. Ask your first yes/no question."}]
    oracle_msgs = []

    n_questions = 0
    n_invalid   = 0
    won         = False
    final_guess = None
    qa_pairs    = []

    if verbose:
        print(f"\n  concept={word!r}  diff={entry['difficulty']:.2f}")

    for _turn in range(MAX_QUESTIONS + 6):
        player_raw  = call_player(player_client, player_msgs)
        player_msgs.append({"role": "assistant", "content": player_raw})
        action_type, action_text = parse_action(player_raw)

        if verbose:
            print(f"    [{n_questions+1:2d}] {action_type:8s}: {action_text!r}")

        if action_type == "guess":
            if re.fullmatch(_PLACEHOLDERS, action_text.strip()):
                action_type = "invalid"
                player_msgs.append({"role": "user",
                    "content": "That wasn't a real guess. Write your actual guess: <guess>word</guess>"})
                n_invalid += 1
                continue
            final_guess = action_text
            won = final_guess.lower().strip() == word.lower().strip()
            if verbose:
                print(f"    -> {'WIN' if won else 'LOSS'}")
            break

        if action_type == "invalid":
            n_invalid += 1
            player_msgs.append({"role": "user",
                "content": "Invalid format. Use <question>your question</question> or <guess>your guess</guess>."})
            if n_invalid >= 3:
                break
            continue

        if n_questions >= MAX_QUESTIONS:
            player_msgs.append({"role": "user",
                "content": "You have used all 20 questions. You must guess now: <guess>your guess</guess>"})
            continue

        n_questions += 1
        oracle_msgs.append({"role": "user",      "content": action_text})
        oracle_answer = call_oracle(oracle_client, oracle_msgs, oracle_sys)
        oracle_msgs.append({"role": "assistant", "content": oracle_answer})
        if verbose:
            print(f"         oracle: {oracle_answer!r}")

        qa_pairs.append({"q": action_text, "a": oracle_answer, "turn": n_questions})
        player_msgs.append({"role": "user", "content": oracle_answer})

    r_val    = reward(won, n_questions)
    pool_n   = 4000   # approximate pool size for optimality gap
    opt_min  = math.ceil(math.log2(max(pool_n, 2)))

    return {
        "word":           word,
        "difficulty":     entry["difficulty"],
        "frequency":      entry.get("frequency"),
        "concreteness":   entry.get("concreteness"),
        "won":            won,
        "final_guess":    final_guess,
        "n_questions":    n_questions,
        "n_invalid":      n_invalid,
        "reward":         r_val,
        "optimality_gap": n_questions - opt_min,
        "qa_pairs":       qa_pairs,
    }


def make_plots(results: list, summary: dict, path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    BG, FG, GREY  = '#0d0d0d', '#ffffff', '#888888'
    WIN_C, LOSS_C = '#69ff47', '#ff4081'

    completed = [r for r in results if "error" not in r]
    if not completed:
        return

    won_mask = [r["won"]         for r in completed]
    qs       = [r["n_questions"] for r in completed]
    rews     = [r["reward"]      for r in completed]
    diffs    = [r["difficulty"]  for r in completed]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor(BG)
    n, wr = summary["n_concepts"], summary["win_rate"]
    fig.suptitle(
        f"20 Questions  |  oracle: {summary['oracle_model']}  "
        f"player: {summary['player_model']}  |  "
        f"n={n}  win={wr:.1%}  avg_q={summary['avg_questions']:.1f}/{MAX_QUESTIONS}",
        color=FG, fontsize=11, fontfamily='monospace', y=0.99)

    for ax in axes.flat:
        ax.set_facecolor(BG)
        for s in ax.spines.values():
            s.set_edgecolor('#333333')
        ax.tick_params(colors=GREY, labelsize=8)
        ax.xaxis.label.set_color(GREY)
        ax.yaxis.label.set_color(GREY)
        ax.title.set_color(FG)

    ax = axes[0, 0]
    bins = list(range(0, MAX_QUESTIONS + 2))
    win_qs  = [q for q, w in zip(qs, won_mask) if w]
    loss_qs = [q for q, w in zip(qs, won_mask) if not w]
    if win_qs:  ax.hist(win_qs,  bins=bins, color=WIN_C,  alpha=0.75, label='Win',  rwidth=0.85)
    if loss_qs: ax.hist(loss_qs, bins=bins, color=LOSS_C, alpha=0.75, label='Loss', rwidth=0.85)
    if qs: ax.axvline(np.mean(qs), color=FG, lw=1.2, ls='--', alpha=0.5,
                      label=f'mean={np.mean(qs):.1f}')
    ax.set_xlabel('Questions used'); ax.set_ylabel('Games')
    ax.set_title('Questions Used per Game')
    ax.legend(fontsize=8, labelcolor=FG, framealpha=0.1)

    ax = axes[0, 1]
    colors = [WIN_C if w else LOSS_C for w in won_mask]
    ax.scatter(range(len(completed)), rews, c=colors, s=35, alpha=0.85, zorder=3)
    if rews: ax.axhline(np.mean(rews), color=FG, lw=1, ls='--', alpha=0.5,
                        label=f"mean={np.mean(rews):.3f}")
    ax.set_ylim(-0.1, 1.65)
    ax.set_xlabel('Game index'); ax.set_ylabel('Reward')
    ax.set_title('Reward per Game')
    ax.legend(fontsize=8, labelcolor=FG, framealpha=0.1)

    ax = axes[1, 0]
    if diffs and len(diffs) >= 4:
        q25, q50, q75 = np.percentile(diffs, [25, 50, 75])
        def qname(d):
            if d <= q25: return 'Q1\n(easy)'
            if d <= q50: return 'Q2'
            if d <= q75: return 'Q3'
            return 'Q4\n(hard)'
        from collections import defaultdict
        qw = defaultdict(list)
        for r in completed:
            qw[qname(r["difficulty"])].append(int(r["won"]))
        labels = ['Q1\n(easy)', 'Q2', 'Q3', 'Q4\n(hard)']
        wrs_q  = [sum(qw.get(l, [0])) / max(len(qw.get(l, [0])), 1) for l in labels]
        counts = [len(qw.get(l, [])) for l in labels]
        ax.bar(labels, wrs_q, color=[WIN_C if w > 0.5 else LOSS_C for w in wrs_q],
               alpha=0.75, width=0.55)
        ax.set_ylim(0, 1.18); ax.set_ylabel('Win rate')
        ax.set_title('Win Rate by Difficulty Quartile')
        for i, (wr_q, cnt) in enumerate(zip(wrs_q, counts)):
            ax.text(i, wr_q + 0.04, f'{wr_q:.0%}\n(n={cnt})',
                    ha='center', color=FG, fontsize=8, fontfamily='monospace')
    else:
        win_count = sum(won_mask)
        ax.bar(['Win', 'Loss'], [win_count, len(completed) - win_count],
               color=[WIN_C, LOSS_C], alpha=0.75, width=0.5)
        ax.set_title('Win / Loss'); ax.set_ylabel('Count')

    ax = axes[1, 1]
    win_d  = [d for d, w in zip(diffs, won_mask) if w]
    win_q  = [q for q, w in zip(qs,    won_mask) if w]
    loss_d = [d for d, w in zip(diffs, won_mask) if not w]
    loss_q = [q for q, w in zip(qs,    won_mask) if not w]
    if win_d:  ax.scatter(win_d,  win_q,  c=WIN_C,  s=40, alpha=0.8, label='Win', zorder=3)
    if loss_d: ax.scatter(loss_d, loss_q, c=LOSS_C, s=40, alpha=0.8,
                          label='Loss', marker='x', linewidths=1.8, zorder=3)
    if diffs and qs:
        z  = np.polyfit(diffs, qs, 1)
        xs = np.linspace(min(diffs), max(diffs), 50)
        ax.plot(xs, np.polyval(z, xs), color=GREY, lw=1, ls='--', alpha=0.6)
    ax.set_xlabel('Concept difficulty (7 - Zipf freq)')
    ax.set_ylabel('Questions used')
    ax.set_title('Questions Used vs Difficulty')
    ax.legend(fontsize=8, labelcolor=FG, framealpha=0.1)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=BG, pad_inches=0.15)
    plt.close()
    print(f"Plots saved -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-concepts", type=int, default=10)
    ap.add_argument("--seed",    type=int,  default=42)
    ap.add_argument("--pool",    default="twenty_questions/pool/pool.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not API_KEY:
        print("ERROR: set API_KEY in the model config section"); return

    random.seed(args.seed)
    pool = load_pool(Path(args.pool))
    print(f"Pool: {len(pool)} concepts loaded from {args.pool}")

    pool_sorted = sorted(pool, key=lambda x: x["difficulty"])
    n    = min(args.n_concepts, len(pool_sorted))
    step = len(pool_sorted) / n
    sample = [pool_sorted[int(i * step)] for i in range(n)]
    random.shuffle(sample)

    print(f"Testing {n} concepts  (seed={args.seed}  oracle={ORACLE_MODEL}  player={PLAYER_MODEL})\n")

    oracle_client = make_client()
    player_client = make_client()

    results = []
    t0 = time.time()

    for i, entry in enumerate(sample):
        print(f"[{i+1:3d}/{n}]  {entry['word']:<22s}  diff={entry['difficulty']:.2f}",
              end="  ", flush=True)
        try:
            r = run_game(oracle_client, player_client, entry, verbose=args.verbose)
            print(f"{'WIN ' if r['won'] else 'LOSS'}  q={r['n_questions']:2d}  reward={r['reward']:.3f}")
            results.append(r)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"word": entry["word"], "difficulty": entry["difficulty"],
                             "won": False, "n_questions": 0, "reward": 0.0,
                             "error": str(e), "qa_pairs": []})

    elapsed = time.time() - t0
    good = [r for r in results if "error" not in r]
    wins = [r for r in good if r["won"]]

    win_rate   = len(wins) / max(len(good), 1)
    avg_q      = sum(r["n_questions"] for r in good) / max(len(good), 1)
    avg_reward = sum(r["reward"]      for r in good) / max(len(good), 1)

    summary = {
        "n_concepts":    n,
        "seed":          args.seed,
        "oracle_model":  ORACLE_MODEL,
        "player_model":  PLAYER_MODEL,
        "win_rate":      round(win_rate, 4),
        "avg_questions": round(avg_q, 2),
        "avg_reward":    round(avg_reward, 4),
        "elapsed_s":     round(elapsed, 1),
        "results":       results,
    }

    print(f"\n{'='*52}")
    print(f"  Win rate:      {win_rate:.1%}  ({len(wins)}/{len(good)})")
    print(f"  Avg questions: {avg_q:.1f} / {MAX_QUESTIONS}")
    print(f"  Avg reward:    {avg_reward:.4f}")
    print(f"  Time:          {elapsed:.0f}s")
    print(f"{'='*52}")

    os.makedirs("test_results", exist_ok=True)
    model_slug = re.sub(r'[^\w.-]', '_', PLAYER_MODEL)
    ts         = datetime.now().strftime("%b%d_%I%M%p")
    json_path  = f"test_results/results_{model_slug}_{ts}.json"
    plot_path  = f"test_results/plots_{model_slug}_{ts}.png"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON  -> {json_path}")
    make_plots(results, summary, plot_path)


if __name__ == "__main__":
    main()
```

---

## Step 6 — `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "twenty-questions"
version = "0.1.0"
description = "20 Questions RL environment for Prime Intellect Hub"
requires-python = ">=3.11"
dependencies = ["verifiers>=0.1.3"]

[project.optional-dependencies]
dev = ["nltk", "wordfreq", "openai", "matplotlib", "numpy"]
```

---

## Step 7 — Run Order

```bash
# 1. install deps
uv pip install nltk wordfreq openai matplotlib numpy
python -c "import nltk; nltk.download('wordnet')"

# 2. build pool (one-time, ~10 seconds)
cd E:\prime-envs\twenty_questions
python pool/build_pool.py --out pool/pool.json --min-freq 2.0

# optional: add concreteness ratings
python pool/build_pool.py --concreteness path/to/concreteness.csv

# 3. sanity check
python -c "import json; p=json.load(open('pool/pool.json')); print(len(p), 'concepts')"

# 4. test (10 concepts)
cd E:\prime-envs
python test_twenty_questions.py --n-concepts 10 --verbose

# 5. scale up
python test_twenty_questions.py --n-concepts 100 --seed 42
```

---

## Pool Schema

```json
{
  "word":         "dog",
  "frequency":    5.43,
  "difficulty":   1.57,
  "concreteness": 4.89
}
```

`concreteness` is omitted if the Brysbaert CSV was not provided.

---

## Result Schema

```json
{
  "n_concepts":    10,
  "seed":          42,
  "oracle_model":  "MiniMax-M2.1",
  "player_model":  "MiniMax-M2.5",
  "win_rate":      0.70,
  "avg_questions": 12.3,
  "avg_reward":    0.815,
  "elapsed_s":     184.5,
  "results": [
    {
      "word":           "dog",
      "frequency":      5.43,
      "difficulty":     1.57,
      "won":            true,
      "final_guess":    "dog",
      "n_questions":    9,
      "n_invalid":      0,
      "reward":         1.275,
      "optimality_gap": 3,
      "qa_pairs": [
        {"q": "Is it a living thing?", "a": "Yes", "turn": 1},
        {"q": "Is it an animal?",      "a": "Yes", "turn": 2}
      ]
    }
  ]
}
```

---

## Diagnostic Plots (4 panels, black background)

| Panel | What to look for |
|---|---|
| Questions Used per Game | wins cluster low (< 12), losses cluster at 20 |
| Reward per Game | win band 1.0–1.5, loss band at 0 |
| Win Rate by Difficulty Quartile | should slope Q1 > Q2 > Q3 > Q4 |
| Questions Used vs Difficulty | positive slope expected |

---

## What Is NOT in v0.1

- No RL training loop (env class exists, not yet wired to a trainer)
- No curriculum sampler (test harness samples evenly by difficulty quartile)
- No Hub upload — pending test validation
- Concreteness ratings optional; if omitted, difficulty = frequency only
