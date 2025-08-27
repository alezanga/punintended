import re
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from utils.io import save_json


def main():
    argparse = ArgumentParser()
    argparse.add_argument('input', type=str)
    argparse.add_argument('output', type=str)
    args = argparse.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    df = pd.read_csv(input_path, sep="\t")

    formatted_samples = list()
    for i, row in df.iterrows():
        _id = row["ID"]
        label = int(re.search(r"^.+_([10])$", _id).group(1))

        structured_pun = {
            "text": row["Joke text"],
            "w_p": None,
            "w_a": None,
            "s_p": None,
            "s_a": None,
            "c_w": None,
            "explanation": None,
            "label": label,
            "is_het": None,
            "id": _id
        }

        if label == 1:
            structured_pun["w_p"] = row["Pun lemma"].strip()
            structured_pun["w_a"] = row["Target lemma"].strip()
            structured_pun["s_p"] = row["Pun synonyms/hypernyms"].strip()
            structured_pun["s_a"] = row["Target synonyms/hypernyms"].strip()
            structured_pun["is_het"] = structured_pun["w_p"] != structured_pun["w_a"]

        formatted_samples.append(structured_pun)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path.with_suffix(".json"), formatted_samples)


if __name__ == "__main__":
    main()
