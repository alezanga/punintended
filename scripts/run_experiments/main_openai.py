import argparse
import json
import logging
import os
import random
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from evaluation.evaluator import Evaluator
from utils.api import call_api
from utils.io import save_json, load_json
from utils.log import setup_logging
from utils.text_processing import preprocessing

LOG_NAME = "llm_oai"

load_dotenv()


def process_dataset(dataset: List[Dict], client: OpenAI, system_prompt: str, user_wrap: str,
                    model_id: str, max_tokens: int, temperature, logger: logging.Logger, merge_results: bool = False) -> None:
    outdir.mkdir(exist_ok=True, parents=True)
    tasks = list()
    for ex in dataset:
        body = call_api(client, system_prompt=system_prompt,
                        user_prompt=user_wrap.format(preprocessing(ex["text"])),
                        model=model_id, max_tokens=max_tokens, return_body=True, temperature=float(temperature))
        task = {
            "custom_id": f"{test_set_name}-{ex['id']}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }
        tasks.append(task)

    file_name = outdir / "batch_tasks.jsonl"
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

    save_json(outdir / "batch_job.json", dict(job_id=batch_job.id, merge_results=merge_results))


def main(model_id: str, dataset: List[Dict[str, Any]], max_tokens: int, temperature: float, prompt: str):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    evaluate_rationale = prompt.endswith("p3.txt") or prompt.endswith("p5.txt")
    system_prompt: str = Path(prompt).read_text()
    user_wrap: str = Path(str(Path(prompt).with_suffix("")) + "-wrap.txt").read_text()

    evaluator = Evaluator(logger, outdir)

    if (outdir / "batch_job.json").exists() and not args.force:
        logger.info(f"**** Evaluate existing results in '{str(output_file)}'")
        if output_file.exists():
            results = load_json(output_file)
            # Get additional results
            existing_ids = {e["id"] for e in results}
            dataset_ids = {e["id"] for e in dataset}
            existing_ids_in_dataset = existing_ids & dataset_ids

            # check the text of existing ids and if they are changed, and if yes, they should be re-evaluated
            results_by_id = {e["id"]: e for e in results}
            for e in dataset:
                example_id = e["id"]
                if example_id in existing_ids_in_dataset:
                    res = results_by_id[example_id]
                    if e["text"] != res["input"]:
                        # should be evaluated
                        existing_ids_in_dataset.remove(example_id)

            ids_to_re_evaluate = dataset_ids - existing_ids_in_dataset

            if ids_to_re_evaluate:
                logger.info(f"**** Re evaluation of {len(ids_to_re_evaluate)} existing results")
                save_json(outdir / "reval.json", list(ids_to_re_evaluate))

                # Evaluate only different ids
                examples_to_re_evaluate = [e for e in dataset if e["id"] in ids_to_re_evaluate]
                # call API only with selected ids
                process_dataset(dataset=examples_to_re_evaluate, client=client, system_prompt=system_prompt, user_wrap=user_wrap,
                                model_id=model_id, max_tokens=max_tokens, temperature=temperature, logger=logger, merge_results=True)
                # Remove the old output file and return, since I will need to get results
                shutil.move(output_file, output_file_old)
                return
            else:
                logger.info("**** Nothing to re-evaluate")

            # import re
            # bias_pattern = re.compile(r"old.*never die.*they|\WTom\W|was only.*daughter.*but|doctor.*doctor|used to.*but|^when the", flags=re.IGNORECASE)
            # non_biased_ids = {e["id"] for e in dataset if not bias_pattern.search(e["text"])}
            # b_len = len(dataset) - len(non_biased_ids)
            # print(f"Removed {b_len}")
            # print(Counter([e["label"] for e in dataset if e["id"] not in non_biased_ids]))
            # # non_biased_ids = random.sample(non_biased_ids, b_len)
            # dataset = [e for e in dataset if e["id"] in non_biased_ids]
            # results = [e for e in results if e["id"] in non_biased_ids]

            # Evaluation
            binary_metrics = evaluator.evaluate_metrics(dataset, results, evaluate_rationale, evaluate_rationale)
            metrics_0 = evaluator.evaluate_metrics(dataset, results, False, evaluate_rationale, pos_class=0, hom_het_evaluation=False, re_parse_results=False)
            binary_metrics["precision_0"] = metrics_0["precision"]
            binary_metrics["recall_0"] = metrics_0["recall"]
            save_json(outdir / "metrics.json", binary_metrics)
            desired_order = ['accuracy', 'f1', 'precision', 'recall']
            if evaluate_rationale:
                desired_order.extend(
                    ['het_recall', 'hom_recall', 'het_support', 'hom_support', 'kw_agreement_2', 'kw_agreement_1',
                     'kw_agreement_0', 'kw_agreement'])
            m = pd.Series(binary_metrics).reindex(desired_order)
            m.to_csv(outdir / "metrics.csv")
        else:
            logger.info("Retrieving using 'get_results'...")
            get_results(dataset, job_id=None)
            if output_file.exists():
                # Evaluate
                main(model_id, dataset, max_tokens, temperature, prompt)
            else:
                logger.error("Error: Output file was not created by 'get_results'. Quitting.")
    else:
        logger.info(f"**** Going from scratch")
        process_dataset(dataset=dataset, client=client, system_prompt=system_prompt, user_wrap=user_wrap,
                        model_id=model_id, max_tokens=max_tokens, temperature=temperature, logger=logger)


def get_results(dataset: List[Dict[str, Any]], job_id: str = None, use_json: bool = False):
    if job_id is None and outdir.exists():
        id_json = load_json(outdir / "batch_job.json")
        job_id = id_json["job_id"]
        merge_with_old_results: bool = id_json.get("merge_results", False)
    else:
        raise FileNotFoundError(f"Job ID file not present or result folder is missing.")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    batch_job = client.batches.retrieve(job_id)

    if batch_job.status == "completed":
        outdir.mkdir(exist_ok=True, parents=True)

        result_file_id = batch_job.output_file_id
        result = client.files.content(result_file_id).content
        result_file_name = outdir / "batch_output.jsonl"
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

        dataset_df = pd.DataFrame(dataset).set_index('id')
        for res in outputs:
            task_id = res['custom_id']
            match = re.match(rf'^{test_set_name}-(.*)$', task_id)
            index = match.group(1)
            example = dataset_df.loc[index].to_dict()
            example["id"] = str(index)
            logprob = None
            logprob_token = None
            try:
                if use_json:
                    result = json.loads(res['response']['body']['choices'][0]['message']['content'])
                else:
                    result = res['response']['body']['choices'][0]['message']['content']
                    logprob = res['response']['body']['choices'][0]['logprobs']['content'][0]['logprob']
                    logprob_token = res['response']['body']['choices'][0]['logprobs']['content'][0]
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"An error occurred: {e}")
                # out = dict(error=str(e))
            else:
                out = {
                    "id": index,
                    "output": result,
                    "confidence": np.exp(logprob) if logprob is not None else None,
                    "confidence_token": logprob_token,
                    "input": example["text"],
                    "label": int(example["label"]),
                    "example": example
                }
                results.append(out)

        if merge_with_old_results:
            results_re_evaluated = results  # new results
            results = load_json(output_file_old)  # old results
            existing_ids = {e["id"] for e in results}  # old ids
            dataset_ids = {e["id"] for e in dataset}  # all the IDs I want (old + new)
            existing_ids_in_dataset = existing_ids & dataset_ids  # old ids I want to keep
            results_by_id = {e["id"]: e for e in results}
            for e in dataset:
                example_id = e["id"]
                if example_id in existing_ids_in_dataset:
                    res = results_by_id[example_id]
                    if e["text"] != res["input"]:
                        # should be evaluated
                        existing_ids_in_dataset.remove(example_id)

            results_from_dataset = [r for r in results if r["id"] in existing_ids_in_dataset]
            results_new = results_from_dataset + results_re_evaluated  # new results + old that must be kept
            logger.info(
                f"Existing {len(existing_ids)} results. Re-evaluated {len(results_re_evaluated)} "
                f"results. Kept {len(results_from_dataset)} results from existing. "
                f"Total is {len(results_new)} results.")
            # overwrite old results with the new merged ones
            results = results_new
        save_json(output_file, results)
    else:
        print(batch_job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="OpenAI model ID")
    parser.add_argument("--prompt", type=str, required=True, help="Path to prompt txt file")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the test data JSON file")
    parser.add_argument("--outdir", type=str, default="results/llm", help="Directory to save results")
    parser.add_argument("--max_tokens", type=int, default=64, help="Maximum tokens for the response")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature for the model")
    parser.add_argument("--position", type=str, choices=['after', 'before'], required=True,
                        help="Ask rationale 'before' or 'after' prompt")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    data = load_json(Path(args.test_data))
    random.shuffle(data)
    test_set_name = Path(args.test_data).with_suffix("").name

    outdir = Path(args.outdir) / test_set_name / Path(args.model_id).name / Path(args.prompt).with_suffix("").name

    output_file = outdir / "output.json"
    output_file_old = output_file.with_suffix(".bak.json")  # name when we need to back up the old results output file

    log_path = outdir
    log_path.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_path, LOG_NAME)
    logger.debug(f"Set up logger in dir '{str(log_path)}'.")

    # HOWTO:
    # 1. call main with desired CLI arguments.
    #   This will schedule the batch and won't return the result
    #   In the result folder a 'batch_job.json' is created with the JOB ID that is needed to retrieve results.
    #
    # 2. Run again to get results and evaluate. If results are not ready, repeat until they are.

    main(args.model_id, data, args.max_tokens, args.temp, args.prompt)
