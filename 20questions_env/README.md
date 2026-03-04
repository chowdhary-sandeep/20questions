# twenty-questions

### Overview
- **Environment ID**: `twenty-questions`
- **Description**: A multi-turn 20 Questions game. An LLM agent identifies a secret English noun by asking up to 20 yes/no questions, answered by an LLM oracle. Wrong guesses count as a turn. Correct guesses end the episode with a reward scaled by efficiency.

### Dataset
- **Pool**: 8,155 English nouns from WordNet, filtered by Brysbaert concreteness ratings and Zipf word frequency
- **Tiers** (percentile-based on difficulty, ascending — tier 1 = easiest):
  - Tier 1 — easiest 1%: ~81 words — baby, eyes, card, road, door, daughter, neck, engine, bill, moon
  - Tier 2 — 1st–5th pct: ~326 words — mother, branch, cotton, diamond, horn, tower, oven, chick, knight, pope
  - Tier 3 — 5th–10th pct: ~408 words — fossil, shorts, flood, canyon, elevator, bulb, coconut, vaccine
  - Tier 4 — 10th–20th pct: ~816 words — rhino, condo, cologne, dairy, electronics, lasagna, swimsuit
  - Tier 5 — rest (> 20th pct): ~6,524 words — machete, psychiatrist, wasabi, alternator, underpass, featherweight

### Task
- **Parser**: `XMLParser` with `<question>` and `<guess>` fields
- **Oracle**: separate LLM that knows the secret word and answers each question with `Yes | No | Sometimes | Unclear`
- **Episode end**: correct guess OR 20 turns exhausted

### Reward
```
correct guess at turn t:  1.0 + 0.5 * (20 - t) / 19   →  [1.0, 1.5]
wrong guess / timeout:    0.0
```
Reward is **sparse and episode-level only** — no per-turn shaping. This makes credit assignment harder and is intentional for RL training signal quality.

### Quickstart
```bash
prime eval run twenty-questions
```

Player and oracle are **independently configurable** — use any model for each:
```bash
# Same model for both (OPENAI_API_KEY used for player and oracle)
prime eval run twenty-questions \
  -m gpt-4.1-mini \
  -a '{"tier": 1, "oracle_model": "gpt-4.1-mini"}'

# Different models: strong oracle, weaker player being trained
prime eval run twenty-questions \
  -m gpt-4o-mini \
  -a '{"tier": 1, "oracle_model": "gpt-4.1"}'

# Local player (vLLM/LM Studio), hosted oracle
prime eval run twenty-questions \
  -m my-local-model \
  -b "http://localhost:8000/v1" \
  -a '{"tier": 2, "oracle_model": "gpt-4.1-mini", "oracle_base_url": "https://api.openai.com/v1"}'
```

### Environment Arguments
| Argument | Type | Default | Description |
|---|---|---|---|
| `tier` | int | `1` | Word difficulty tier (1–5): 1=easiest (~81 words), 5=hardest (~6,524 words) |
| `oracle_model` | str | `"gpt-4.1-mini"` | Oracle LLM — answers yes/no questions. Independent of player model (`-m`). |
| `oracle_base_url` | str | `"https://api.openai.com/v1"` | Oracle API base URL (use any OpenAI-compatible endpoint) |
| `oracle_api_key_var` | str | `"OPENAI_API_KEY"` | Env var holding the oracle API key |
| `num_train_examples` | int | `2000` | Words sampled for training |
| `num_eval_examples` | int | `50` | Words sampled for evaluation |
| `system_prompt` | str | `DEFAULT_SYSTEM_PROMPT` | System prompt for the player agent |
| `seed` | int | `0` | Random seed |

### Metrics
| Metric | Meaning |
|---|---|
| `reward` | Episode reward (0.0 or 1.0–1.5) |
| `win_rate` | Fraction of episodes with correct guess |
| `avg_questions` | Average turns used per episode |
| `efficiency_bonus` | Average `reward - 1.0` for wins (0 = used all 20 questions, 0.5 = guessed on turn 1) |

### See Also
[MiniMax plays 20 Questions](https://huggingface.co/spaces/echoboi/minimax2-1-plays-20-questions)
