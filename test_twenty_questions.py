"""
Twenty Questions — Two-model test harness
Oracle:  MiniMax (knows the concept + ConceptNet facts)
Player:  MiniMax (clean context, no concept knowledge)

Usage:
    python test_twenty_questions.py                    # 10 concepts
    python test_twenty_questions.py --n-concepts 50
    python test_twenty_questions.py --n-concepts 100 --seed 99
    python test_twenty_questions.py --verbose          # print Q&A live
"""

import argparse, json, math, os, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ── import env (mirrors a Hub-downloaded package) ────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "twenty_questions"))
from twenty_questions import (
    load_pool, parse_action, reward, is_correct_guess,
    oracle_system_prompt, PLAYER_SYSTEM_PROMPT, MAX_QUESTIONS,
)

# ── Model config — edit here to swap backend ─────────────────────────────────
from openai import OpenAI

# Local (LM Studio) — inactive
# API_KEY      = "lm-studio"
# BASE_URL     = "http://127.0.0.1:1234/v1"
# ORACLE_MODEL = "openai/gpt-oss-20b"
# PLAYER_MODEL = "openai/gpt-oss-20b"

# MiniMax — active
API_KEY      = os.environ.get("MINIMAX_API_KEY", "")
BASE_URL     = "https://api.minimaxi.chat/v1"
ORACLE_MODEL = "MiniMax-M2.1"   # fast, non-reasoning — just Yes/No
PLAYER_MODEL = "MiniMax-M2.5"   # reasoning model for strategic questioning

def make_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Oracle needs thinking ON to reason correctly; player has it OFF for speed
_ORACLE_EXTRA = {}
_PLAYER_EXTRA = {}
_PLACEHOLDERS = re.compile(
    r'[.\s]*|piano|word|your answer here|your guess|your_guess', re.IGNORECASE)  # thinking disabled

# ── single oracle call ────────────────────────────────────────────────────────
_ORACLE_KEYWORDS = ("yes", "no", "sometimes", "unclear")

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
    # Find first keyword anywhere in the response (word-boundary match)
    m = re.search(r'\b(yes|no|sometimes|unclear)\b', content, re.IGNORECASE)
    return m.group(1).capitalize() if m else "Unclear"

# ── single player call ────────────────────────────────────────────────────────
def call_player(player_client, messages: list) -> str:
    resp = player_client.chat.completions.create(
        model=PLAYER_MODEL,
        messages=[{"role": "system", "content": PLAYER_SYSTEM_PROMPT}] + messages,
        temperature=0.7,
        max_tokens=2000,
        **_PLAYER_EXTRA,
    )
    content = resp.choices[0].message.content or ""
    # strip Qwen3 <think>...</think> block if present
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return content

# ── single game ───────────────────────────────────────────────────────────────
def run_game(oracle_client, player_client,
             entry: dict,
             verbose: bool = False) -> dict:

    concept    = entry["word"]
    oracle_sys = oracle_system_prompt(concept)

    # MiniMax requires at least one user message; seed with game-start prompt
    player_msgs = [{"role": "user", "content": "The game begins. Ask your first yes/no question."}]
    oracle_msgs = []   # oracle sees: system + [user(question), assistant(answer), ...]

    n_questions = 0
    n_invalid   = 0
    won         = False
    final_guess = None
    qa_pairs    = []

    if verbose:
        print(f"\n  word={concept!r}  diff={entry['difficulty']:.2f}")

    for _turn in range(MAX_QUESTIONS + 6):
        # ── player acts ───────────────────────────────────────────────────────
        player_raw = call_player(player_client, player_msgs)
        player_msgs.append({"role": "assistant", "content": player_raw})
        action_type, action_text = parse_action(player_raw)

        if verbose:
            print(f"    [{n_questions+1:2d}] {action_type:8s}: {action_text!r}")

        # ── reject placeholder guesses — treat as invalid ────────────────────
        if action_type == "guess":
            if re.fullmatch(_PLACEHOLDERS, action_text.strip()):
                action_type = "invalid"
                player_msgs.append({"role": "user",
                    "content": "That wasn't a real guess. Use <guess>word</guess>."})
                n_invalid += 1
                continue

        # ── invalid format ────────────────────────────────────────────────────
        if action_type == "invalid":
            n_invalid += 1
            player_msgs.append({"role": "user",
                "content": "Invalid format. Use <question>...</question> or <guess>...</guess>."})
            if n_invalid >= 3:
                break
            continue

        # ── out of questions — force a guess ──────────────────────────────────
        if n_questions >= MAX_QUESTIONS:
            player_msgs.append({"role": "user",
                "content": "You have used all 20 questions. Make your final guess: <guess>word</guess>"})
            continue

        # ── correct guess → WIN ───────────────────────────────────────────────
        if action_type == "guess" and is_correct_guess(action_text, concept):
            final_guess = action_text
            won = True
            if verbose:
                print(f"    -> WIN")
            break

        # ── wrong guess OR question → ask oracle ──────────────────────────────
        # A wrong guess is treated as a question: "Is it X?" → "No, that's not it."
        if action_type == "guess":
            oracle_q    = f"Is it {action_text}?"
            final_guess = action_text
            oracle_msgs.append({"role": "user", "content": oracle_q})
            oracle_answer = "No"
            oracle_msgs.append({"role": "assistant", "content": oracle_answer})
            if verbose:
                print(f"         oracle: 'No'  (wrong guess)")
        else:
            oracle_q = action_text
            oracle_msgs.append({"role": "user", "content": oracle_q})
            oracle_answer = call_oracle(oracle_client, oracle_msgs, oracle_sys)
            oracle_msgs.append({"role": "assistant", "content": oracle_answer})
            if verbose:
                print(f"         oracle: {oracle_answer!r}")

        n_questions += 1
        qa_pairs.append({"q": oracle_q, "a": oracle_answer, "turn": n_questions})
        player_msgs.append({"role": "user", "content": oracle_answer})

    pool_n  = 13415   # approximate pool size for optimality gap
    opt_min = math.ceil(math.log2(max(pool_n, 2)))
    r_val   = reward(won, n_questions)

    return {
        "word":           concept,
        "difficulty":     entry["difficulty"],
        "frequency":      entry.get("frequency"),
        "won":            won,
        "final_guess":    final_guess,
        "n_questions":    n_questions,
        "n_invalid":      n_invalid,
        "reward":         r_val,
        "optimality_gap": n_questions - opt_min,
        "qa_pairs":       qa_pairs,
    }

# ── plots ─────────────────────────────────────────────────────────────────────
def make_plots(results: list, summary: dict, path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    BG, FG, GREY  = '#0d0d0d', '#ffffff', '#888888'
    WIN_C, LOSS_C = '#69ff47', '#ff4081'

    completed = [r for r in results if "error" not in r]
    if not completed:
        print("No completed games to plot.")
        return

    won_mask = [r["won"]         for r in completed]
    qs       = [r["n_questions"] for r in completed]
    rews     = [r["reward"]      for r in completed]
    diffs    = [r["difficulty"]  for r in completed]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor(BG)
    n, wr = summary["n_concepts"], summary["win_rate"]
    avg_q = summary["avg_questions"]
    fig.suptitle(
        f"20 Questions  |  oracle: {summary['oracle_model']}  player: {summary['player_model']}  |  "
        f"n={n}  win={wr:.1%}  avg_q={avg_q:.1f}/{MAX_QUESTIONS}",
        color=FG, fontsize=11, fontfamily='monospace', y=0.99)

    for ax in axes.flat:
        ax.set_facecolor(BG)
        for s in ax.spines.values():
            s.set_edgecolor('#333333')
        ax.tick_params(colors=GREY, labelsize=8)
        ax.xaxis.label.set_color(GREY)
        ax.yaxis.label.set_color(GREY)
        ax.title.set_color(FG)

    # 1 — question count histogram
    ax = axes[0, 0]
    bins = list(range(0, MAX_QUESTIONS + 2))
    win_qs  = [q for q, w in zip(qs, won_mask) if w]
    loss_qs = [q for q, w in zip(qs, won_mask) if not w]
    if win_qs:
        ax.hist(win_qs,  bins=bins, color=WIN_C,  alpha=0.75, label='Win',  rwidth=0.85)
    if loss_qs:
        ax.hist(loss_qs, bins=bins, color=LOSS_C, alpha=0.75, label='Loss', rwidth=0.85)
    if qs:
        ax.axvline(np.mean(qs), color=FG, lw=1.2, ls='--', alpha=0.5,
                   label=f'mean={np.mean(qs):.1f}')
    ax.set_xlabel('Questions used')
    ax.set_ylabel('Games')
    ax.set_title('Questions Used per Game')
    ax.legend(fontsize=8, labelcolor=FG, framealpha=0.1)

    # 2 — reward scatter
    ax = axes[0, 1]
    colors = [WIN_C if w else LOSS_C for w in won_mask]
    ax.scatter(range(len(completed)), rews, c=colors, s=35, alpha=0.85, zorder=3)
    if rews:
        ax.axhline(np.mean(rews), color=FG, lw=1, ls='--', alpha=0.5,
                   label=f"mean={np.mean(rews):.3f}")
    ax.set_ylim(-0.1, 1.65)
    ax.set_xlabel('Game index')
    ax.set_ylabel('Reward')
    ax.set_title('Reward per Game')
    ax.legend(fontsize=8, labelcolor=FG, framealpha=0.1)

    # 3 — win rate by difficulty quartile
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
        bar_colors = [WIN_C if w > 0.5 else LOSS_C for w in wrs_q]
        ax.bar(labels, wrs_q, color=bar_colors, alpha=0.75, width=0.55)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel('Win rate')
        ax.set_title('Win Rate by Difficulty Quartile')
        for i, (wr_q, cnt) in enumerate(zip(wrs_q, counts)):
            ax.text(i, wr_q + 0.04, f'{wr_q:.0%}\n(n={cnt})',
                    ha='center', color=FG, fontsize=8, fontfamily='monospace')
    else:
        win_count = sum(won_mask)
        ax.bar(['Win', 'Loss'], [win_count, len(completed) - win_count],
               color=[WIN_C, LOSS_C], alpha=0.75, width=0.5)
        ax.set_title('Win / Loss')
        ax.set_ylabel('Count')

    # 4 — questions vs difficulty scatter
    ax = axes[1, 1]
    win_d  = [d for d, w in zip(diffs, won_mask) if w]
    win_q  = [q for q, w in zip(qs,    won_mask) if w]
    loss_d = [d for d, w in zip(diffs, won_mask) if not w]
    loss_q = [q for q, w in zip(qs,    won_mask) if not w]
    if win_d:
        ax.scatter(win_d,  win_q,  c=WIN_C,  s=40, alpha=0.8, label='Win', zorder=3)
    if loss_d:
        ax.scatter(loss_d, loss_q, c=LOSS_C, s=40, alpha=0.8,
                   label='Loss', marker='x', linewidths=1.8, zorder=3)
    if len(diffs) >= 3:
        z = np.polyfit(diffs, qs, 1)
        xs = np.linspace(min(diffs), max(diffs), 50)
        ax.plot(xs, np.polyval(z, xs), color=GREY, lw=1, ls='--', alpha=0.6)
    ax.set_xlabel('Concept difficulty score')
    ax.set_ylabel('Questions used')
    ax.set_title('Questions Used vs Difficulty')
    ax.legend(fontsize=8, labelcolor=FG, framealpha=0.1)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=BG, pad_inches=0.15)
    plt.close()
    print(f"Plots saved -> {path}")

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-concepts", type=int, default=10)
    ap.add_argument("--seed",    type=int,  default=42)
    ap.add_argument("--pool",    default="twenty_questions/pool/pool_enriched.json")
    ap.add_argument("--tier",    type=int,  default=1,
                    help="1=concrete(>=4.0)  2=moderate(3-4)  3=borderline(2.5-3)")
    ap.add_argument("--concepts", nargs="+", metavar="WORD",
                    help="Pin specific concepts to test (skips random sampling)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not API_KEY:
        print("ERROR: set API_KEY in the model config section")
        sys.exit(1)

    random.seed(args.seed)
    pool = load_pool(Path(args.pool))
    print(f"Pool: {len(pool)} words loaded from {args.pool}")

    if args.concepts:
        pool_index = {e["word"].lower(): e for e in pool}
        sample = []
        for w in args.concepts:
            entry = pool_index.get(w.lower())
            if entry is None:
                # not in pool — create a minimal entry so the game still runs
                entry = {"word": w, "difficulty": 0.0, "concreteness": 5.0}
            sample.append(entry)
        n = len(sample)
    else:
        # tier filter
        tier_ranges = {1: (4.0, 9.9), 2: (3.0, 4.0), 3: (2.5, 3.0)}
        lo, hi = tier_ranges.get(args.tier, (0, 9.9))
        tier_pool = [e for e in pool if lo <= e.get("concreteness", 0) < hi]
        print(f"Tier {args.tier} (concreteness {lo}-{hi}): {len(tier_pool)} words")
        n = min(args.n_concepts, len(tier_pool))
        sample = random.sample(tier_pool, n)

    model_prefix = PLAYER_MODEL.split('/')[-1][:3].lower()
    print(f"Testing {n} words  (seed={args.seed}  oracle={ORACLE_MODEL}  player={PLAYER_MODEL})\n")

    os.makedirs("test_results", exist_ok=True)
    results = []
    t0 = time.time()
    print_lock = __import__('threading').Lock()

    def run_one(i, entry):
        # each game gets its own client pair (thread-safe)
        oc = make_client()
        pc = make_client()
        try:
            r = run_game(oc, pc, entry, verbose=args.verbose)
            tag = "WIN " if r["won"] else "LOSS"
            with print_lock:
                print(f"[{i+1:3d}/{n}]  {entry['word']:<22s}  "
                      f"conc={entry.get('concreteness',0):.2f}  "
                      f"diff={entry['difficulty']:.3f}  "
                      f"{tag}  q={r['n_questions']:2d}  reward={r['reward']:.3f}")
            json_path = f"test_results/results_{entry['word']}_{model_prefix}.json"
            with open(json_path, "w") as f:
                json.dump(r, f, indent=2)
            return r
        except Exception as e:
            with print_lock:
                print(f"[{i+1:3d}/{n}]  {entry['word']:<22s}  ERROR: {e}")
            return {"word": entry["word"], "difficulty": entry["difficulty"],
                    "won": False, "n_questions": 0, "reward": 0.0,
                    "error": str(e), "qa_pairs": []}

    with ThreadPoolExecutor(max_workers=min(n, 16)) as pool:
        futures = {pool.submit(run_one, i, entry): entry for i, entry in enumerate(sample)}
        for fut in as_completed(futures):
            results.append(fut.result())

    elapsed = time.time() - t0
    good = [r for r in results if "error" not in r]
    wins = [r for r in good if r["won"]]

    win_rate   = len(wins) / max(len(good), 1)
    avg_q      = sum(r["n_questions"] for r in good) / max(len(good), 1)
    avg_reward = sum(r["reward"]      for r in good) / max(len(good), 1)

    print(f"\n{'='*52}")
    print(f"  Win rate:      {win_rate:.1%}  ({len(wins)}/{len(good)})")
    print(f"  Avg questions: {avg_q:.1f} / {MAX_QUESTIONS}")
    print(f"  Avg reward:    {avg_reward:.4f}")
    print(f"  Time:          {elapsed:.0f}s")
    print(f"{'='*52}")
    print(f"\nResults -> test_results/results_{{word}}_{model_prefix}.json  (one per word)")

if __name__ == "__main__":
    main()
