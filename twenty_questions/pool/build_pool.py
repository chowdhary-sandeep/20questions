"""
WordNet nouns + wordfreq -> pool.json

Output schema per concept:
{
  "word":         "dog",
  "frequency":    5.43,      # Zipf scale (0=rare, 7=very common)
  "difficulty":   1.57,      # 7.0 - frequency  (higher = harder)
  "concreteness": 4.89       # Brysbaert 1-5 rating (omitted if CSV not found)
}

Run:
  python pool/build_pool.py
  python pool/build_pool.py --min-freq 2.5 --max-freq 5.3
  python pool/build_pool.py --concreteness path/to/concreteness.csv
"""

import argparse, json, os
from pathlib import Path


def is_common_noun(word: str) -> bool:
    """
    True if the word is primarily a common (non-proper) noun.
    Rejects:
      - words with no noun synsets
      - words where ALL noun synsets are named entities (proper nouns)
      - words where more verb or adjective synsets exist than noun synsets
    """
    from nltk.corpus import wordnet as wn
    noun_synsets = wn.synsets(word, pos='n')
    if not noun_synsets:
        return False
    # reject if every noun sense is a named entity
    if all(s.instance_hypernyms() for s in noun_synsets):
        return False
    n = len(noun_synsets)
    v = len(wn.synsets(word, pos='v'))
    a = len(wn.synsets(word, pos='a')) + len(wn.synsets(word, pos='s'))
    return n >= v and n >= a


def build_pool(min_freq: float = 2.5,
               max_freq: float = 5.3,
               min_len:  int   = 4,
               concreteness_csv: str | None = None) -> list:
    from nltk.corpus import wordnet as wn
    from wordfreq import zipf_frequency

    print("Extracting WordNet nouns…")
    candidates: set[str] = set()
    for synset in wn.all_synsets('n'):
        for lemma in synset.lemmas():
            word = lemma.name().lower()
            if '_' not in word and word.isalpha() and len(word) >= min_len:
                candidates.add(word)
    print(f"  {len(candidates):,} candidate single-word nouns")

    # Brysbaert concreteness ratings (optional enrichment)
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

    print(f"Filtering: common-noun check + freq [{min_freq}, {max_freq}]…")
    pool = []
    for word in candidates:
        freq = zipf_frequency(word, 'en')
        if freq < min_freq or freq > max_freq:
            continue
        if not is_common_noun(word):
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
    lo = pool[0]["difficulty"]  if pool else 0
    hi = pool[-1]["difficulty"] if pool else 0
    print(f"  Pool: {len(pool):,} concepts  (difficulty {lo:.2f} – {hi:.2f})")
    return pool


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",           default=str(Path(__file__).parent / "pool.json"))
    ap.add_argument("--min-freq",      type=float, default=2.5)
    ap.add_argument("--max-freq",      type=float, default=5.3)
    ap.add_argument("--min-len",       type=int,   default=4)
    ap.add_argument("--concreteness",  default=None,
                    help="Path to Brysbaert concreteness CSV (optional)")
    args = ap.parse_args()

    pool = build_pool(
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        min_len=args.min_len,
        concreteness_csv=args.concreteness,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(pool, f, indent=2)
    print(f"Saved {len(pool):,} concepts -> {args.out}")
