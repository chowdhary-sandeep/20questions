# 20 Questions RL Environment — Plan

## Motivation

20 Questions is one of the purest instantiations of active information seeking. An agent must identify a hidden target by asking yes/no questions, choosing each to maximally reduce uncertainty. Optimal strategy: bisect the hypothesis space each turn, ~1 bit per question, log₂(N) questions for N possibilities.

Current LLMs are bad at this. They ask redundant questions, fail to track what's been eliminated, and don't exploit hierarchical concept structure. This is a *reasoning strategy* gap, not a knowledge gap. We propose the first online RL training environment for 20Q on the Prime Intellect Environments Hub.

The core skill — narrowing an uncertain search space through strategically chosen queries — is the same skill needed for scientific inquiry, where an agent designs experiments to disambiguate competing hypotheses about hidden laws.

---

## Prior Work

Early RL approaches used custom (non-LLM) agents on fixed knowledge bases. Hu et al. (2018, EMNLP — retracted) trained an MLP/LSTM policy on 1K famous people with 500 predefined questions. Key contribution: a learned RewardNet for credit assignment against sparse episode-end reward. A follow-up extended this to ConceptNet/WordNet knowledge bases (128–250 concepts), finding performance degraded with sparse feature matrices.

Bertolazzi et al. (2023, INLG) evaluated ChatGPT on tiny concept sets (8–16 items from McRae feature norms). ChatGPT asks low-EIG, redundant questions and only approaches optimal play when explicitly prompted to list remaining hypotheses — outsourcing Bayesian bookkeeping to the prompt. Mazzaccara et al. (2024, EMNLP Findings) improved questioning quality via DPO on EIG-ranked preference pairs sampled from Llama-2-7B, using McRae norms and BigBench concepts. Transfer across domains worked. But DPO is offline — no environment interaction, no rollouts, no exploration.

BIG-Bench (2022) included 20Q self-play on a small fixed concept list. AR-Bench (Zhou et al., 2025, ICML) tested ~12 frontier LLMs on active reasoning tasks (detective cases, puzzles, number guessing) and found all models struggle — tree search and post-training yield only modest gains. Will Brown's Wordle environment on Prime Intellect proved multi-turn GRPO training works for small models (Qwen3-0.6B, 1.7B) on sequential reasoning games, trainable on few GPUs in hours.

**The gap:** No 20Q environment for online RL training of LLMs. Prior work is either RL with non-LLM agents, or LLM evaluation without training. Mazzaccara's DPO is closest but remains offline. The Wordle env proves the infrastructure works; 20Q is the natural next step. All prior work used tiny concept sets (8–541 items from McRae/BigBench norms) because they needed feature matrices for EIG computation. We don't — our oracle is an LLM and our reward is outcome-only. This unlocks scaling to thousands of concepts trivially.

---

## Connection to Scientific Discovery

This environment is the first stage of a curriculum toward training agents for scientific discovery — reverse-engineering hidden rules from observations.

In 20Q, the concept space has natural metric structure. Concepts cluster by shared properties, and a question like "is it alive?" cleanly partitions the space along a meaningful boundary. Each answer eliminates a well-defined subtree. The mapping from questions to hypothesis-space reductions is direct and approximately tree-structured.

In physics, this breaks down. The agent observes dynamical trajectories and must infer the hidden rule generating them. Two very different rules can produce nearly identical dynamics (degeneracy), and two similar rules can produce wildly different dynamics (sensitivity). The map from observations to hypotheses has no guaranteed Lipschitz continuity — you cannot bisect the hypothesis space with clean categorical questions. The agent must design experiments that disambiguate hypotheses equivalent under typical observations but divergent under carefully constructed ones.

Yet, similarities can be observed. In both cases the agent faces a hidden target and must reduce uncertainty through sequential queries. The core competency is identical: maintain an implicit model of what remains possible, choose actions that maximally shrink that space. In the end, you are narrowing the search space by asking the right questions and trying to do so cleverly.

---

## Environment Specification

### Structure

```
Agent (questioner) ←→ Oracle (LLM, always)
                         ↑
                    Target word (hidden)
```

Multi-turn, single-player. GRPO-compatible: episode-level reward, no per-step rewards needed.

### Concept Pool — Common English Nouns + Brysbaert Concreteness

The oracle is an LLM — it already knows what every word means. We just need a clean word list with a difficulty proxy. No feature matrices, no knowledge graphs. Prior work used tiny pools (8–541 items from McRae/BigBench norms) because they needed feature matrices for EIG computation. We don't — this unlocks scaling trivially.

**Pipeline (see `enrich_pool.py`):**

1. **Extract nouns:** All single-word English nouns from NLTK WordNet, filtered to `len > 2`, alphabetic only.
2. **Get frequency:** `wordfreq` library (Zipf scale 0–7, aggregates SUBTLEX/Wikipedia/Reddit). Filter `freq >= 2.0` to remove obscure words. Yields ~13K nouns (`pool.json`).
3. **Join with Brysbaert concreteness:** 40K English words rated 1–5 by 4,000+ humans (Brysbaert et al., 2014). Inner join on exact word match → 76.8% coverage (10,299 words).
4. **Filter:** Remove words with concreteness < 2.5 (too abstract for yes/no questions: "peace", "quality", "culture"). Yields **~8,200 playable words**.
5. **Difficulty score:** `difficulty = 0.4 × freq_difficulty + 0.6 × conc_difficulty`. Concreteness weighted higher — a rare concrete word (haystack) is guessable, a common abstract word (quality) is not.

**Result — three tiers:**

| Tier | Concreteness | Count | Examples |
|------|-------------|-------|---------|
| 1 (concrete) | ≥ 4.0 | ~4,200 | spoon, drum, oyster, ballerina |
| 2 (moderate) | 3.0–4.0 | ~2,800 | conference, cortex, scripture |
| 3 (borderline) | 2.5–3.0 | ~1,200 | poltergeist, buffer, excursion |

**Curriculum:** v0.1 trains on Tier 1 only. v0.2 adds Tier 2. Tier 3 optional. Sampling shifts from easy (high-frequency, high-concreteness) → uniform over training.

### Oracle

Always an LLM. Receives the target word and the agent's question, responds Yes/No/Sometimes. "Sometimes" handles genuinely ambiguous cases (e.g., "Is a tomato a vegetable?" — sometimes). This is more realistic than forcing binary answers and teaches the agent to reason about partial evidence.

```
System prompt to oracle:
"The secret word is '{word}'. Answer the player's yes/no question
 truthfully and concisely. Reply only 'Yes', 'No', or 'Sometimes'."
```

Oracle consistency: cache (question, answer) pairs within each game session so the same question always gets the same answer.

### Action Space

Agent output is free-form text, parsed into:
- `<question>Is it a living thing?</question>` → oracle answers Yes/No/Sometimes
- `<guess>elephant</guess>` → checked against target, game ends

### Reward Function (GRPO-compatible, episode-level)

```python
def reward(correct: bool, n_turns: int, max_turns: int = 20) -> float:
    if correct:
        return 1.0 + 0.5 * (max_turns - n_turns) / max_turns
    else:
        return 0.0
```

Episode-level only. GRPO samples N rollouts per target word, ranks by reward, uses relative advantage for the policy gradient. No per-step rewards, no EIG in the reward. The agent discovers efficient questioning from outcome signal alone.

### Core Loop

```
1. Sample target word from pool (difficulty-weighted)
2. Agent receives system prompt + rules
3. For turn t = 1..20:
   a. Agent outputs <question> or <guess>
   b. If question → oracle LLM responds Yes/No/Sometimes
   c. If guess → exact string match against target, end game
4. If 20 turns exhausted without correct guess → reward = 0
5. Return episode reward
```

### Verifiers Integration

```python
class TwentyQuestionsEnv(Environment):
    def load_dataset(self):
        return self.pool  # list of {'word': ..., 'difficulty': ...}
    
    def get_system_prompt(self, example):
        return "You are playing 20 Questions. Identify the secret word by asking yes/no questions. " \
               "Use <question>...</question> to ask or <guess>...</guess> to guess."
    
    def rubric(self, trajectory) -> float:
        correct = trajectory.final_guess == trajectory.target
        return 1.0 + 0.5 * (20 - trajectory.n_turns) / 20 if correct else 0.0
```

---

## Implementation Priorities

**v0.1 — MVP (1-2 days)**
- [ ] Word pool: use `pool_enriched.json` (8,213 words), train on Tier 1 only (~4,200)
- [ ] LLM oracle (Yes/No/Sometimes) with answer caching per session
- [ ] Reward: correctness + efficiency (episode-level, GRPO-compatible)
- [ ] Verifiers env structure, test with Qwen3-0.6B/1.7B
- [ ] Eval: zero-shot win rate for 3-4 frontier models

**v0.2 — Curriculum + diagnostics**
- [ ] Add Tier 2 concepts (~7,000 total), difficulty-based sampling scheduler
- [ ] Log diagnostic metrics: questions-to-win distribution, win rate vs difficulty band

**v0.3 — Robustness**
- [ ] Multi-word targets, proper nouns, ambiguous words
- [ ] Oracle consistency hardening (structured fact prompts)
- [ ] Partial match handling (synonyms, plurals)

**v0.4 — Toward discovery env**
- [ ] Structured rule targets (not single words)
- [ ] Bridge to Discovery Environment

---

## Open Design Questions

- **Tier 3 curation**: The 2.5–3.0 concreteness band is noisy. Some words work (poltergeist, excursion), many don't (issuer, readability). May need manual review or skip entirely.
- **Oracle consistency**: LLM may contradict itself across questions. Caching helps within-session. Across sessions (same word, same question), we want determinism — seed the oracle or use temperature=0.
- **Guess matching**: Exact string match is brittle. "Elephant" vs "elephants" vs "an elephant" — use lemmatization? Embedding similarity threshold?
- **Prior distribution**: Uniform over pool? Frequency-weighted (more common words appear more often)? Uniform is cleaner for GRPO — same difficulty distribution across rollout groups.
- **Multi-agent variant**: Two LLMs play against each other. One thinks, one guesses. Could train both.
