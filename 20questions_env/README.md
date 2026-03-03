# twenty-questions

### Overview
- **Environment ID**: `twenty-questions`
- **Description**: A multi-turn 20 Questions game. An LLM agent identifies a secret English noun by asking up to 20 yes/no questions, answered by an LLM oracle. Wrong guesses count as a turn ("No, that's not it"). Correct guesses end the episode with a reward scaled by efficiency.
- **Tags**: multi-turn, game, reasoning, language, train, eval

### Dataset
- **Pool**: 8,213 English nouns from WordNet, filtered by Brysbaert concreteness ratings (Brysbaert et al., 2014) and Zipf word frequency (wordfreq)
- **Tiers**:
  - Tier 1 — concrete (concreteness ≥ 4.0): ~4,200 words (dog, spoon, volcano, penicillin)
  - Tier 2 — moderate (3.0–4.0): ~2,800 words
  - Tier 3 — borderline abstract (2.5–3.0): ~1,200 words
- **Difficulty score**: `0.4 * (7 - zipf_freq) / 5 + 0.6 * (5 - concreteness) / 4`

### Task
- **Type**: Multi-turn, adversarial game
- **Parser**: `XMLParser` with `<question>` and `<guess>` fields
- **Oracle**: LLM (configurable, default: `gpt-4.1-mini`) receives the secret word and answers each question with `Yes | No | Sometimes | Unclear`
- **Episode end**: correct guess OR 20 turns exhausted

### Reward
```
correct guess at turn t:  1.0 + 0.5 * (20 - t) / 20   →  [1.0, 1.5]
wrong guess / timeout:    0.0
```
Reward is **sparse and episode-level only** — no per-turn shaping. This makes credit assignment harder and is intentional for RL training signal quality.

### Quickstart
```bash
prime eval run twenty-questions
```

With args:
```bash
prime eval run twenty-questions \
  -m gpt-4.1-mini \
  -n 50 -r 1 \
  -a '{"tier": 1, "oracle_model": "gpt-4.1-mini"}'
```

### Required Environment Variables
| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | API key for the oracle model (default) |

### Environment Arguments
| Argument | Type | Default | Description |
|---|---|---|---|
| `tier` | int | `1` | Word difficulty tier: 1=concrete, 2=moderate, 3=abstract |
| `oracle_model` | str | `"gpt-4.1-mini"` | Model used to answer questions |
| `oracle_base_url` | str | `"https://api.openai.com/v1"` | Oracle API base URL |
| `oracle_api_key_var` | str | `"OPENAI_API_KEY"` | Env var holding the oracle API key |
| `num_train_examples` | int | `2000` | Words sampled for training |
| `num_eval_examples` | int | `50` | Words sampled for evaluation |
| `seed` | int | `0` | Random seed |

### Metrics
| Metric | Meaning |
|---|---|
| `reward` | Episode reward (0.0 or 1.0–1.5) |
| `win_rate` | Fraction of episodes with correct guess |
| `avg_questions` | Average turns used per episode |
| `efficiency_bonus` | Average `reward - 1.0` for wins (0 = used all 20 questions, 0.5 = guessed on turn 1) |

### Changelog
#### v0.1.0
- Initial release
- Pool: 8,213 WordNet nouns enriched with Brysbaert concreteness ratings
- LLM oracle with Yes/No/Sometimes/Unclear responses
- Wrong guesses treated as questions (count toward 20-turn limit)
- Sparse episode-level reward only
- WordNet synonym/hyponym matching for correct-guess evaluation
