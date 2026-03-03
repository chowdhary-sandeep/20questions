# twenty-questions vs. Existing Prime Intellect Environments

## TL;DR Summary

The `twenty-questions` environment is **closest to `will/wordle`** structurally and in training target, but occupies a distinct niche across all comparison dimensions. No existing env on the hub trains the same skill (active information seeking via open-ended yes/no questioning) with online RL.


| Dimension        | Closest Match                     | 20Q's Differentiator                                                                       |
| ---------------- | --------------------------------- | ------------------------------------------------------------------------------------------ |
| **Conceptual**   | Wordle                            | Free-form NL questions vs. constrained guesses; ConceptNet curriculum                      |
| **Reward**       | Wordle (correctness + efficiency) | Sparse outcome-only — NO per-turn shaping, no partial credit, EIG is diagnostic not reward |
| **Rollout**      | Balrog-prime (long multi-turn)    | LLM oracle with ConceptNet grounding; up to 20 turns; no deterministic game engine         |
| **Target agent** | Wordle (small LLM + GRPO)         | Same — but 20Q's harder credit assignment is the challenge                                 |


The most important gap 20Q fills: **no env on Prime Intellect trains an LLM to discover efficient hypothesis-space partitioning from outcome signal alone**. Everything close either (a) provides rich per-turn feedback (Wordle), (b) constrains the action space to tools/structured calls (wiki-search, deepdive), or (c) is eval-only / offline training (reasoning-core).

---

## Compared Environments

Environments selected by conceptual proximity to 20Q:

1. `**will/wordle`** — closest structural match (sequential guessing game, multi-turn, small LLMs via GRPO)
2. `**will/wiki-search**` / `**primeintellect/wiki-search**` — active info seeking via tool queries
3. `**primeintellect/deepdive**` — web-scale agentic search with redundancy penalty
4. `**sileod/reasoning-core-env**` — symbolic reasoning / math (same "reasoning" bucket)
5. `**sangeeth/balrog-prime**` — long-horizon multi-turn games for LLMs

---

## 1. twenty-questions vs. will/wordle

### Conceptual Similarity: VERY HIGH

Both are single-player sequential guessing games where the agent must identify a hidden target through structured feedback. Both train hypothesis refinement and elimination. Both are designed for small LLM training via GRPO on Prime Intellect infrastructure.

**Key structural difference**: Wordle's action space is a fixed dictionary of 5-letter English words. The agent's strategy is constrained to letter position feedback (green/yellow/gray). 20Q's action space is open-ended natural language — the agent can ask anything, and the hypothesis space is not explicitly bounded (the agent does not see the ConceptNet pool). Wordle is combinatorially constrained. 20Q is epistemically unconstrained.

### Reward Construction


|                      | Wordle                                   | twenty-questions                     |
| -------------------- | ---------------------------------------- | ------------------------------------ |
| **Type**             | Multi-component, dense                   | Outcome-only, sparse                 |
| **Per-turn signal**  | Yes — 0.2×greens + 0.1×yellows per guess | No — reward is 0 until game ends     |
| **Correctness**      | 1.0 if exact match                       | 1.0 if correct guess                 |
| **Efficiency bonus** | 1/num_guesses if correct                 | +0.5×(max_q − n_q)/max_q if correct  |
| **Format penalty**   | Yes (0.2 weight)                         | TBD                                  |
| **EIG/info signal**  | None (game gives color code)             | Computed post-hoc as diagnostic only |


Wordle's dense per-turn feedback (color codes) significantly eases the credit assignment problem. The agent knows immediately whether its guess was useful. 20Q deliberately removes this — the agent gets no signal until it wins or runs out of questions. This is harder to train but produces a more general skill (there is no analog of "color code" in physics hypothesis discovery).

Wordle's `partial_answer` reward (greens + yellows) is essentially a hand-crafted information gain proxy. 20Q explicitly rejects this design choice: the agent must learn what makes a question good from outcomes alone.

### Rollout Structure


|                     | Wordle                              | twenty-questions                                |
| ------------------- | ----------------------------------- | ----------------------------------------------- |
| **Max turns**       | 6 (Wordle rules)                    | 20                                              |
| **Action**          | Free text → parsed to 5-letter word | Free text → parsed to `<question>` or `<guess>` |
| **Oracle**          | Deterministic (exact letter match)  | LLM with ConceptNet ground-truth prompt         |
| **State**           | Letter feedback accumulates         | Oracle answers accumulate (agent's own context) |
| **Done condition**  | Correct guess or 6 turns            | Correct guess or 20 questions                   |
| **Pool visibility** | Agent doesn't know pool             | Agent doesn't know pool                         |


Oracle reliability: Wordle's oracle is deterministic and infallible. 20Q's LLM oracle can be inconsistent. Mitigation: include ConceptNet property edges in oracle system prompt as ground truth. Cache answers within episode.

### Target Agent

Both target small LLMs (Qwen3-0.6B, 1.7B) trained via GRPO on Prime Intellect infrastructure. Both use the verifiers framework. Wordle has demonstrated this works (referenced in the 20Q plan). 20Q's harder credit assignment (no per-turn signal) may require more rollouts per concept to get a training signal.

### Unique to 20Q vs. Wordle

- **Continuous difficulty curriculum** (ConceptNet IsA depth + siblings + property count) vs. Wordle's fixed word list
- **Abstract concepts** (emotions, ideas) with qualitatively different strategy requirements — Wordle has no analog
- **EIG diagnostic logging** — Wordle tracks nothing about questioning quality, 20Q logs realized IG, optimality gap, entropy trajectory
- **LLM oracle** — opens the door to ambiguous/noisy oracles in v0.3+
- **Bridge to Discovery Env** — Wordle has no planned extension to scientific reasoning

---

## 2. twenty-questions vs. will/wiki-search & primeintellect/wiki-search

### Conceptual Similarity: MEDIUM-HIGH

Both train an agent to actively search for information about a hidden target through multi-turn interaction. But the search mechanisms differ fundamentally:

- **Wiki-search**: Agent issues *tool calls* (`search_pages`, `view_sections`, `read_section`) to a fixed corpus (Wikipedia). The target is a factual answer. The agent's strategy is about *navigation and retrieval* from a document corpus.
- **20Q**: Agent issues *natural language questions* to an LLM oracle. The target is a concept identity. The agent's strategy is about *hypothesis space partitioning*.

In wiki-search, success requires finding the right document and extracting the answer. In 20Q, success requires formulating questions that eliminate categories of concepts efficiently.

### Reward Construction


|                         | Wiki-search                                                   | twenty-questions                           |
| ----------------------- | ------------------------------------------------------------- | ------------------------------------------ |
| **Type**                | LLM-judge, binary                                             | Outcome-only, sparse                       |
| **Judge**               | External LLM (GPT-4.1-mini) evaluates coherence + correctness | No judge — rule-based oracle check         |
| **Efficiency**          | None — no penalty for extra tool calls                        | Yes — efficiency bonus for fewer questions |
| **Partial credit**      | None                                                          | None                                       |
| **Intermediate signal** | None                                                          | None                                       |


Wiki-search's LLM judge introduces variance (judge can disagree with itself) and latency (extra API call per episode). 20Q's oracle check is rule-based (string match against concept name) — cheaper and deterministic.

Critically: wiki-search does not penalize tool usage. An agent can call `search_pages` 10 times and get the same reward as one call. 20Q penalizes via the efficiency component: every extra question reduces the reward ceiling.

### Rollout Structure


|                    | Wiki-search                                  | twenty-questions                       |
| ------------------ | -------------------------------------------- | -------------------------------------- |
| **Max turns**      | 10                                           | 20                                     |
| **Action type**    | Structured tool call                         | Natural language question or guess     |
| **Oracle**         | ChromaDB embedding search + markdown parsing | LLM with ConceptNet grounding          |
| **Corpus**         | Fixed Wikipedia subset                       | ConceptNet pool (agent doesn't see it) |
| **Done condition** | Agent calls `finish` or 10 turns             | Correct guess or 20 questions          |


Wiki-search's action space is structured (JSON tool calls). 20Q's is free-form. This means wiki-search is easier to parse reliably but cannot ask free-form questions that don't correspond to pre-defined tool signatures.

### Target Agent

Wiki-search is designed for medium-to-large models with reliable tool-use capability. 20Q targets small models where tool-call parsing is less reliable — hence free-form questions with simple XML parsing.

### Unique to 20Q vs. Wiki-search

- **Information-theoretic framing**: 20Q can measure EIG per question; wiki-search has no analog of hypothesis space reduction
- **No external API costs**: 20Q's oracle is a self-hosted LLM; wiki-search requires embedding search infrastructure
- **Curriculum**: 20Q difficulty is concept-grounded; wiki-search has no curriculum
- **Efficiency penalty**: 20Q explicitly penalizes extra questions; wiki-search doesn't

---

## 3. twenty-questions vs. primeintellect/deepdive

### Conceptual Similarity: MEDIUM

DeepDive trains an agent to search the live web (via Serper API) and synthesize answers from heterogeneous sources. Both 20Q and DeepDive are about reducing uncertainty through strategic queries, but at different scales and structures.

DeepDive's uncertainty is *epistemic about facts* (where is the answer in the web?). 20Q's uncertainty is *conceptual* (which concept is the target?). The former requires synthesis from multiple sources; the latter requires sequential binary partitioning.

### Reward Construction


|                         | DeepDive                                                                | twenty-questions                           |
| ----------------------- | ----------------------------------------------------------------------- | ------------------------------------------ |
| **Type**                | Judge + redundancy penalty                                              | Correctness + efficiency bonus             |
| **Correctness**         | LLM judge (GPT-4o level)                                                | Rule-based oracle check                    |
| **Efficiency signal**   | Soft penalty for repeated queries (Jaccard similarity between searches) | Hard bonus: fewer questions = higher score |
| **Intermediate signal** | None                                                                    | None                                       |
| **Cost**                | Expensive (Serper API + judge LLM)                                      | Cheap (one oracle LLM)                     |


DeepDive's redundancy penalty is a *soft* efficiency incentive computed via Jaccard similarity of search queries. 20Q's efficiency component is *hard*: the agent's total reward is mathematically tied to question count. DeepDive can still get full reward even with redundant searches if the judge deems the answer correct; 20Q's max reward decreases monotonically with questions asked.

### Rollout Structure


|                 | DeepDive                                   | twenty-questions                     |
| --------------- | ------------------------------------------ | ------------------------------------ |
| **Max turns**   | 32                                         | 20                                   |
| **Action type** | Structured tool calls (search, scan, open) | Natural language questions / guesses |
| **Oracle**      | Real web (Serper API + HTML parsing)       | LLM with ConceptNet grounding        |
| **State**       | URL/document cache                         | Oracle answer history (in context)   |


DeepDive's oracle is the real world — results can be wrong, outdated, or contradictory. This is realistic but introduces noise that is hard to control during training. 20Q's oracle is constrained by ConceptNet ground truth in the system prompt, making it more controllable.

### Target Agent

DeepDive targets frontier models (GPT-4o class) capable of complex multi-step tool orchestration. 20Q targets small models where simplicity of oracle interaction (yes/no questions) enables training with limited compute.

---

## 4. twenty-questions vs. sileod/reasoning-core-env

### Conceptual Similarity: LOW

Reasoning-core is a single-turn benchmark/training env for symbolic and mathematical reasoning tasks. The agent receives a problem and outputs an answer in one turn. There is no interactive search, no oracle, no multi-turn state.

The conceptual distance from 20Q is large: reasoning-core tests *static reasoning ability* (can the model solve this problem?), while 20Q tests *dynamic information seeking* (can the model ask the right questions to identify something?).

Reasoning-core is relevant as a contrast class: it represents what 20Q is *not*. If you just need a math benchmark, reasoning-core exists. 20Q is about the active reasoning process *before* a problem is fully specified.

### Reward Construction


|                | Reasoning-core               | twenty-questions                 |
| -------------- | ---------------------------- | -------------------------------- |
| **Type**       | Task-specific scorer, binary | Correctness + efficiency, sparse |
| **Feedback**   | None (single turn)           | None until end                   |
| **Efficiency** | None                         | Yes (fewer questions)            |
| **Curriculum** | Procedural task generator    | ConceptNet difficulty score      |


Reasoning-core delegates scoring to `reasoning_core.score_answer()` — a library function that handles different task types (exact match, numerical tolerance, etc.). 20Q's reward is simple and self-contained.

### Rollout Structure


|                     | Reasoning-core     | twenty-questions                       |
| ------------------- | ------------------ | -------------------------------------- |
| **Turns**           | 1                  | Up to 20                               |
| **Action**          | Single text output | Sequential questions + final guess     |
| **State evolution** | None               | Accumulating oracle answers in context |


Single-turn vs. multi-turn is the core structural difference. Reasoning-core cannot train sequential decision-making.

---

## 5. twenty-questions vs. sangeeth/balrog-prime

### Conceptual Similarity: MEDIUM

Balrog-prime wraps long-horizon game environments (NetHack, BabyAI, TextWorld, Crafter) for LLM training. Both 20Q and Balrog-prime are multi-turn environments where the agent must take sequential actions to reach a goal. Both are designed for RL training on LLMs.

The structural similarity is at the infrastructure level: multi-turn rollouts, LLM policy, outcome reward. The semantic difference is large: Balrog-prime's agent navigates a game world; 20Q's agent navigates a hypothesis space.

### Reward Construction


|                  | Balrog-prime                                      | twenty-questions                          |
| ---------------- | ------------------------------------------------- | ----------------------------------------- |
| **Type**         | Modular: success / progress / efficiency / hybrid | Fixed: correctness + efficiency bonus     |
| **Main signal**  | Sparse success (1.0 if true termination)          | Sparse correctness (1.0 if correct guess) |
| **Shaping**      | Optional: progress = normalized episode return    | None — deliberate choice                  |
| **Efficiency**   | Optional: 1 - (steps/max_steps)                   | Yes — built into reward formula           |
| **Configurable** | Yes — rubric_weights dict                         | No — reward structure is fixed            |


Balrog-prime offers *optional reward shaping* (progress and efficiency signals) to ease credit assignment. 20Q deliberately disables shaping: the agent gets zero intermediate signal. This is not a limitation but a design choice — the goal is to train a skill that transfers to settings where shaping signals (like EIG) are unavailable.

### Rollout Structure


|                           | Balrog-prime                               | twenty-questions                      |
| ------------------------- | ------------------------------------------ | ------------------------------------- |
| **Max turns**             | Up to 200                                  | 20                                    |
| **Action**                | Game-specific (NLE actions, text commands) | Natural language questions / guesses  |
| **Oracle**                | Real game engine (NLE, BabyAI, etc.)       | LLM with ConceptNet grounding         |
| **Partial observability** | Yes (game fog of war, etc.)                | Yes (agent doesn't know concept pool) |
| **State**                 | Game state (map, inventory, etc.)          | Oracle answers in context             |


Balrog-prime's state is rich and structured (game observations). 20Q's state is minimal (conversation history of oracle answers). This makes 20Q more accessible to small models: the agent doesn't need to parse complex game states, just track which questions got "yes" vs. "no".

### Target Agent

Both target LLMs with RL training (GRPO or similar). Balrog-prime includes VLM support (images from games). 20Q is text-only. Balrog-prime's episodes can be 200 turns — significantly longer than 20Q's 20, increasing training cost per episode.

### Unique to 20Q vs. Balrog-prime

- **Hypothsis-space framing**: 20Q's agent is explicitly narrowing a concept space; Balrog-prime's agent is navigating a game world
- **Oracle type**: 20Q uses LLM oracle (flexible, generalizable); Balrog-prime uses game engine (deterministic, game-specific)
- **Episode length**: 20Q has fixed 20-turn horizon; Balrog-prime varies (10–200 per game)
- **Knowledge grounding**: 20Q uses ConceptNet as internal ground truth for diagnostic EIG; Balrog-prime has no such diagnostic layer
- **Curriculum**: 20Q has principled difficulty curriculum from ConceptNet; Balrog-prime selects from available game environments

---

## What 20Q Does That No Existing Env Does

### 1. Online RL for Active Hypothesis Search (not tool use, not game play)

All multi-turn environments on Prime Intellect either:

- Use structured tool calls (wiki-search, deepdive, acebench)
- Play a constrained game with bounded action space (wordle, lights-out, 2048)
- Navigate a game world (balrog-prime)

20Q is the only environment where the agent's task is to reduce uncertainty about a hidden target by asking free-form natural language questions — the core structure of scientific inquiry, diagnostic reasoning, and hypothesis-driven search.

### 2. Outcome-Only Reward with EIG as Diagnostic (not reward)

No existing env separates these two roles:

- Wordle bakes in information gain via color code feedback
- DeepDive uses redundancy penalty as a soft info-efficiency signal
- Wiki-search has no efficiency signal at all

20Q computes EIG internally but never passes it to the agent. This is what makes the learned skill potentially transferable to settings (physics discovery) where EIG is not computable.

### 3. ConceptNet-Grounded Continuous Difficulty Curriculum

No other env derives difficulty from a knowledge graph structure (IsA depth × sibling density × property richness). This enables:

- Smooth difficulty interpolation during training
- Agent never knows its difficulty — open-ended world model
- Abstract concepts emerge naturally at the upper difficulty range

### 4. Designed as Stage 1 of a Scientific Discovery Curriculum

20Q is the only env explicitly positioned as a precursor to open-ended hypothesis discovery. The reward structure (outcome-only, no EIG shaping) is specifically designed to produce a transferable skill, not just a game-optimal policy.

---

## Open Comparison Questions

- **Will small models (Qwen3-0.6B) learn from pure outcome reward?** Wordle shows GRPO works for multi-turn games, but Wordle's per-turn signal makes credit assignment tractable. 20Q's 20-turn horizon with end-only reward is harder. May need larger initial models or a warmup phase.
- **How inconsistent will the LLM oracle be?** ConceptNet grounding in the system prompt mitigates this but doesn't eliminate it. Worth testing oracle consistency before full training runs.
- **Does EIG improve as a training signal?** The plan claims it shouldn't be in the reward. But it's worth running an ablation: train one model with EIG reward, one without, compare transfer to a held-out concept domain.

