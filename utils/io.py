import json
import os
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import yaml


def save_or_append_txt(save_file, np_outputs):
    """
    Save or append numpy array to a text file.

    :param save_file: Path to the text file where the data will be saved.
    :param np_outputs: Numpy array to be saved.
    """
    # Check if the file exists
    file_exists = os.path.isfile(save_file)

    # Use 'a' mode to append if the file exists, otherwise use 'w' mode
    with open(save_file, 'a' if file_exists else 'w') as f:
        np.savetxt(f, np_outputs, fmt="%i")


def save_json(file_path: Path, data: Union[Dict, List]) -> None:
    """
    Save a dictionary as a JSON file at the specified file path.

    :param file_path: Path object representing the file path where the JSON will be saved.
    :param data: Dictionary object to be saved as JSON.
    """
    # Write the dictionary to a JSON file
    with file_path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=True, indent=4)


def load_json(file_path: Path) -> Union[Dict, List]:
    with file_path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def remove_duplicates(file_path: Path) -> None:
    """
    Load JSON data from a file, remove duplicate entries based on 'input' and 'label',
    and overwrite the original file with the deduplicated data.

    :param file_path: Path object representing the file path where the JSON data is stored.
    """
    # Load existing data from the JSON file
    with file_path.open(mode="r", encoding="utf-8") as fo:
        existing_data = json.load(fo)

    # Use a set to track seen (input, label) pairs
    seen = set()
    deduplicated_data = list()

    for entry in existing_data:
        # Create a tuple of the fields to check for duplicates
        key = (entry['input'], entry['label'])

        if key not in seen:
            seen.add(key)
            deduplicated_data.append(entry)

    # Overwrite the original file with the deduplicated data
    with file_path.open(mode="w", encoding="utf-8") as fo:
        json.dump(deduplicated_data, fo, ensure_ascii=True, indent=4)


def load_yaml(path: Union[str, Path]) -> dict:
    """
    Load YAML as python dict

    @param path: path to YAML file
    @return: dictionary containing data
    """
    with open(path, encoding="UTF-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data


def dump_yaml(data, path: Union[str, Path]) -> None:
    """
    Load YAML as python dict

    @param path: path to YAML file
    @param data: data to dump
    @return: dictionary containing data
    """
    with open(path, encoding="UTF-8", mode="w") as f:
        yaml.dump(data, f, Dumper=yaml.SafeDumper)
