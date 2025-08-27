from pathlib import Path
from typing import Dict, List

from datasets import Dataset, NamedSplit

from utils.text_processing import preprocessing


def format_chat_dataset(examples: List[Dict], add_rationale: bool, extended_rationale: bool, system_prompt: str,
                        user_wrap: str,
                        dump_folder: Path, split: NamedSplit) -> Dataset:
    formatted_chat = list()
    for example in examples:
        label = example["label"]
        text = preprocessing(example["text"])
        if label == 1:
            output_text = "yes"
            if add_rationale:
                w_p = preprocessing(example["w_p"])
                w_a = preprocessing(example["w_a"])
                output_text += f" <{w_p}> <{w_a}>"
            if extended_rationale:
                s_p = preprocessing(example["s_p"])
                s_a = preprocessing(example["s_a"])
                output_text += f" <{s_p}> <{s_a}>"
        else:
            output_text = "no"
            if add_rationale:
                output_text += f" <> <>"
            if extended_rationale:
                output_text += f" <> <>"

        formatted_chat.append({
            "chat": [
                {
                    "from": "system",
                    "value": system_prompt
                }, {
                    "from": "user",
                    "value": user_wrap.format(text)
                }, {
                    "from": "assistant",
                    "value": output_text
                }
            ]
        })
    dataset = Dataset.from_list(formatted_chat, split=split)  # features=features_schema
    dataset.save_to_disk(dump_folder)
    return dataset


def apply_template(tokenizer, examples):
    messages = examples["chat"]
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}
