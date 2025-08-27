import argparse
import json
import os
import re
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv
from openai import OpenAI

from utils.api import call_api
from utils.io import save_json, load_json

load_dotenv()

JSONL = True


def get_prompt() -> Tuple[str, str]:
    wrap = Path("dataset_generation/wrap.txt").read_text()
    if JSONL:
        return Path("dataset_generation/puns_generation.txt").read_text(), wrap
    else:
        return Path("dataset_generation/bad_puns.txt").read_text(), wrap


def clean_string(input_string):
    # Replace spaces with underscores
    modified_string = input_string.replace(" ", "_")
    # Remove all characters that are not letters or numbers
    cleaned_string = re.sub(r'[^a-zA-Z0-9_]', '', modified_string)
    return cleaned_string


def main(model_id: str):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    sys_prompt, wrap = get_prompt()

    structures = load_json(Path("dataset_generation/structures.json"))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    tasks = list()
    for iid, expr in enumerate(structures):
        id_name = clean_string(expr)
        user_prompt = '{"expression": "' + expr + '", "count": ' + str(args.num) + '}'
        body = call_api(client, user_prompt=wrap.format(user_prompt), system_prompt=sys_prompt,
                        model=model_id, max_tokens=args.max_tokens, return_body=True,
                        temperature=1.0)
        task = {
            "custom_id": f"task-{id_name}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }
        tasks.append(task)

    file_name = results_dir / "batch_tasks.jsonl"
    with open(file_name, "w") as file:
        for obj in tasks:
            file.write(json.dumps(obj) + "\n")

    batch_file = client.files.create(
        file=open(file_name, "rb"),
        purpose="batch"
    )
    print(batch_file)

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    batch_job = client.batches.retrieve(batch_job.id)
    print(batch_job)

    save_json(results_dir / "batch_job.json", dict(job_id=batch_job.id))


def get_results():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    results_dir = Path(args.results_dir)
    if results_dir.exists():
        job_id = load_json(results_dir / "batch_job.json")["job_id"]
    else:
        raise FileNotFoundError(f"Job ID file not present or result folder is missing.")

    batch_job = client.batches.retrieve(job_id)

    if batch_job.status == "completed":
        output_file = results_dir / "output.json"
        result_file_name = results_dir / "batch_output.jsonl"

        result_file_id = batch_job.output_file_id
        result = client.files.content(result_file_id).content
        with open(result_file_name, "wb") as file:
            file.write(result)

        # Loading data from saved file
        results = list()
        outputs = list()
        with open(result_file_name, 'r') as file:
            for line in file:
                # Parsing the JSON string into a dict and appending to the list of results
                json_object = json.loads(line.strip())
                outputs.append(json_object)

        for res in outputs:
            try:
                out = {"output": res['response']['body']['choices'][0]['message']['content'],
                       "custom_id": res["custom_id"]}
            except (json.JSONDecodeError, KeyError) as e:
                print(f"An error occurred: {e}")
            else:
                results.append(out)
        save_json(output_file, results)

        for res in results:
            id_name = res["custom_id"]
            if id_name.startswith("task-"):
                id_name = id_name[5:]
            else:
                raise ValueError(f"ID wrongly formatted: {id_name}")
            generations_for_expr = [s.strip() for s in res["output"].split("\n")]
            if JSONL:
                puns = list()
                non_puns = list()
                output_file = results_dir / id_name
                output_file.mkdir(exist_ok=True, parents=True)
                for s in generations_for_expr:
                    try:
                        json_s = json.loads(s)
                    except json.JSONDecodeError as e:
                        print(f"An error occurred: {e}")
                        print(s)
                    else:
                        pun = json_s["pun"]
                        non_pun = json_s["non-pun"]
                        puns.append(pun)
                        non_puns.append(non_pun)
                save_json(output_file / "puns.json", puns)
                save_json(output_file / "non_puns.json", non_puns)
            else:
                output_file = results_dir / f"{id_name}_output.json"
                save_json(output_file, generations_for_expr)
    else:
        print(batch_job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--num", type=int, default=10)
    args = parser.parse_args()

    # HOWTO:
    # 1. call main with desired CLI arguments.
    #   This will schedule the batch and won't return the result
    #   In the result folder a 'batch_job.json' is created with the JOB ID that is needed to retrieve results.
    #
    # 2. comment main and de-comment get_results, pasting the JOB ID in the parameter 'job_id'
    #   If not finished it will print the status
    #   If it finishes it will create a bunch of files in the result dir.
    #   You can repeatedly call this function to see when it finished.
    #
    # 3. When step 2 has finished run main function again (comment get_results) to get metrics.

    main(args.model_id)
    # get_results()
