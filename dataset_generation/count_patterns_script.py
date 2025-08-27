import re
from pathlib import Path

import pandas as pd
import spacy

from utils.io import load_json

nlp = spacy.load('en_core_web_lg')

datasets = {
    "PunEval (train)": "data/public/puneval/train.json",
    "PunEval (test)": "data/public/puneval/test.json",
    "PunEval (val)": "data/public/puneval/val.json",
    "NAP": "data/public/nap.json",
    "JOKER": "data/private/joker/ruined.json",
    "Unbiased": ["data/public/punny_pattern/daughter.json", "data/public/punny_pattern/doctor.json", "data/public/punny_pattern/never_die.json",
                 "data/public/punny_pattern/tom.json",
                 "data/public/punny_pattern/used.json", "data/public/punny_pattern/when.json"],
    "Substitution": "data/public/pun_break.json"
}

patterns = {
    "never_die": re.compile(r"old.*never die.*they", re.IGNORECASE),
    "tom": re.compile(r"\bTom\b"),
    "when": re.compile(r"^when the", re.IGNORECASE),
    "daughter": re.compile(r"she was only.*daughter.{0,2}but", re.IGNORECASE),
    "doctor": re.compile(r"doctor.{0,2}doctor", re.IGNORECASE),
    "used": re.compile(r"used to.*but", re.IGNORECASE)
}

frequencies = list()
statistics = list()
for dataset_name, dataset_path in datasets.items():
    dataset_samples = list()
    if isinstance(dataset_path, list):
        dataset_path = [Path(d) for d in dataset_path]
        for d in dataset_path:
            dataset_samples.extend(load_json(d))
    else:
        dataset_path = Path(dataset_path)
        dataset_samples = load_json(dataset_path)

    # DATA STATISTICS
    total_chars = sum(len(sample["text"]) for sample in dataset_samples)
    total_words = sum(len(nlp(sample["text"])) for sample in dataset_samples)
    avg_chars = total_chars / len(dataset_samples)
    avg_words = total_words / len(dataset_samples)
    het_count = sum((True == sample["is_het"] and sample["label"] == 1) for sample in dataset_samples)
    hom_count = sum((False == sample["is_het"] and sample["label"] == 1) for sample in dataset_samples)
    pun_count = sum(sample["label"] == 1 for sample in dataset_samples)
    assert pun_count == het_count + hom_count, "Wrong number of puns."
    non_pun_count = sum(sample["label"] == 0 for sample in dataset_samples)
    assert len(dataset_samples) == pun_count + non_pun_count, "Wrong number of samples."
    statistics.append((dataset_name, avg_chars, avg_words, het_count, hom_count, non_pun_count, len(dataset_samples)))

    # PATTERN COUNT
    for pattern_name, pattern in patterns.items():
        pattern_occurrences = sum(pattern.search(sample["text"]) is not None for sample in dataset_samples)
        frequencies.append((dataset_name, pattern_name, pattern_occurrences))

df_pattern_count = pd.DataFrame(frequencies, columns=["dataset", "pattern", "occurrences"])
s = df_pattern_count.to_string(index=False)
print(s)

df_statistics = (pd.DataFrame(statistics, columns=["dataset", "avg_chars", "avg_words", "het_count", "hom_count", "non_pun_count", "total_count"])
                 .round()
                 .astype({"avg_chars": int, "avg_words": int, "het_count": int, "hom_count": int, "non_pun_count": int, "total_count": int}))
s = df_statistics.to_string(index=False)
print(s)
