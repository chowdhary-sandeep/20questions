"""
Enrich 20Q word pool with Brysbaert concreteness ratings.

INPUT:  pool.json — list of {word, frequency, difficulty} from WordNet nouns + wordfreq
OUTPUT: pool_enriched.json — filtered to concreteness >= 2.5, with combined difficulty score

STEPS:
1. Clone Brysbaert concreteness ratings from GitHub (40K English words, 1-5 scale)
2. Inner join pool.json words with Brysbaert on exact word match (76.8% match rate)
3. Compute combined difficulty = 0.4 * freq_difficulty + 0.6 * concreteness_difficulty
   - freq_difficulty = (7.0 - zipf_frequency) / 5.0   → 0 = common, 1 = rare
   - conc_difficulty = (5.0 - concreteness) / 4.0      → 0 = concrete, 1 = abstract
   - Concreteness weighted higher: a rare concrete word (haystack) is still guessable,
     a common abstract word (quality) is not.
4. Filter out words with concreteness < 2.5 (too abstract for yes/no questions)
5. Sort by difficulty ascending

RESULT: ~8,200 playable words in three tiers:
  - Tier 1 (conc >= 4.0): ~4,200 words — concrete objects, animals, foods
  - Tier 2 (conc 3.0-4.0): ~2,800 words — moderate abstraction
  - Tier 3 (conc 2.5-3.0): ~1,200 words — borderline, some noise

DATA SOURCE:
  Brysbaert, M., Warriner, A.B., & Kuperman, V. (2014).
  "Concreteness ratings for 40 thousand generally known English word lemmas."
  Behavior Research Methods, 46(3), 904-911.
  GitHub: https://github.com/ArtsEngine/concreteness
  File: Concreteness_ratings_Brysbaert_et_al_BRM.txt (tab-separated)
  Columns used: Word, Conc.M (mean concreteness, 1-5 scale)
"""

import json
import csv
import os
import subprocess

# --- Config ---
POOL_INPUT = "pool.json"
POOL_OUTPUT = "pool_enriched.json"
BRYSBAERT_REPO = "https://github.com/ArtsEngine/concreteness.git"
BRYSBAERT_FILE = "concreteness/Concreteness_ratings_Brysbaert_et_al_BRM.txt"
CONCRETENESS_CUTOFF = 2.5  # exclude words below this

# --- Step 1: Get Brysbaert data ---
if not os.path.exists(BRYSBAERT_FILE):
    subprocess.run(["git", "clone", BRYSBAERT_REPO], check=True)

# --- Step 2: Load Brysbaert concreteness ratings ---
brysbaert = {}
with open(BRYSBAERT_FILE, "r") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        word = row["Word"].strip().lower()
        try:
            brysbaert[word] = float(row["Conc.M"])
        except (ValueError, KeyError):
            pass

print(f"Brysbaert loaded: {len(brysbaert)} words")

# --- Step 3: Load pool.json ---
with open(POOL_INPUT) as f:
    pool = json.load(f)

print(f"Pool loaded: {len(pool)} words")

# --- Step 4: Inner join on exact word match ---
enriched = []
for item in pool:
    word = item["word"]
    if word in brysbaert:
        conc = brysbaert[word]
        freq = item["frequency"]

        # Combined difficulty score
        freq_diff = (7.0 - freq) / 5.0        # 0 = common, 1 = rare
        conc_diff = (5.0 - conc) / 4.0         # 0 = concrete, 1 = abstract
        difficulty = 0.4 * freq_diff + 0.6 * conc_diff

        enriched.append({
            "word": word,
            "frequency": freq,
            "concreteness": round(conc, 2),
            "difficulty": round(difficulty, 3),
        })

print(f"Matched: {len(enriched)} / {len(pool)} ({len(enriched)/len(pool)*100:.1f}%)")

# --- Step 5: Filter out too-abstract words ---
playable = [e for e in enriched if e["concreteness"] >= CONCRETENESS_CUTOFF]
playable.sort(key=lambda x: x["difficulty"])

print(f"After concreteness filter (>= {CONCRETENESS_CUTOFF}): {len(playable)} words")

# --- Step 6: Save ---
with open(POOL_OUTPUT, "w") as f:
    json.dump(playable, f, indent=2)

print(f"Saved to {POOL_OUTPUT}")

# --- Summary ---
tiers = {
    "Tier 1 (concrete >= 4.0)": [e for e in playable if e["concreteness"] >= 4.0],
    "Tier 2 (moderate 3.0-4.0)": [e for e in playable if 3.0 <= e["concreteness"] < 4.0],
    "Tier 3 (borderline 2.5-3.0)": [e for e in playable if 2.5 <= e["concreteness"] < 3.0],
}
for name, words in tiers.items():
    print(f"  {name}: {len(words)} words")
