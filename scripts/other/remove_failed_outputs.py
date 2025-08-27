import os
import re
from pathlib import Path

from utils.io import load_json, save_json


def clean_output_json(directory):
    # Define the regex pattern
    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "output.json":
                file_path = Path(root, file)
                data = load_json(file_path)

                clean = list()
                wrong = list()

                splits = str(file_path).split(os.sep)
                require_rationale = "p3" in splits or "p5" in splits
                pat = r"\b\W{,2}(yes|no)\W{,2}\b(?:[^<>]{,5}<([^>]*)>)" + ("+" if require_rationale else "*")
                pattern = re.compile(pat, flags=re.IGNORECASE)

                # Process each object in the list
                for obj in data:
                    if 'output' in obj and 'id' in obj:
                        output_text = obj['output']
                        if isinstance(output_text, str) and pattern.findall(output_text.strip()):
                            clean.append(obj)
                        else:
                            wrong.append((obj['id'], output_text))

                # If there are wrong outputs, prompt the user
                if wrong:
                    print(f"\n********************* File: {file_path}")
                    print(f"*** The following {len(wrong)} IDs have wrong outputs:")
                    print([_id for _id, _ in wrong])

                    user_input = input("Do you want to delete these objects? (Y/N): ").strip().lower()
                    if user_input == 'y':
                        # Overwrite the file with the clean list
                        save_json(file_path, clean)
                        print(f"Deleted wrong outputs from {file_path}.")
                    else:
                        print(f"*** Kept all objects in {file_path}.")


if __name__ == "__main__":
    start_directory = input("Enter the start directory: ").strip()
    clean_output_json(start_directory)
