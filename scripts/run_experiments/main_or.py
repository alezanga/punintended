import asyncio
import json
import os
import random
import shutil
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

import aiofiles
import aiohttp
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from evaluation.evaluator import Evaluator
from utils.io import save_json, load_json
from utils.log import setup_logging
from utils.text_processing import preprocessing

LOG_NAME = "llm_or"

load_dotenv()


async def save_results_to_disk(output_file, results, lock: asyncio.Lock):
    async with lock:  # Only one task can write to the file at a time
        async with aiofiles.open(output_file, mode="w") as f:
            await f.write(json.dumps(results, indent=4))


async def fetch(logger, session, url, headers, body, retries=5, timeout=20):
    timeout_obj = aiohttp.ClientTimeout(total=timeout)  # Set the total timeout
    for attempt in range(retries):
        try:
            async with session.post(url, headers=headers, json=body, timeout=timeout_obj) as response:
                response.raise_for_status()  # Raise an error for bad responses
                return await response.json()  # Return the JSON response
        except (aiohttp.ClientError, aiohttp.http_exceptions.HttpProcessingError, asyncio.TimeoutError) as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            logger.error(traceback.format_exc())
            if attempt == retries - 1:
                return None  # Return None after max retries
    return None


async def process_request(logger, model_id: str, sys_prompt: str, text: str, session, timeout: int, base_url: str, api_key_variable: str, **kwargs):
    headers = {
        "Authorization": f"Bearer {os.environ.get(api_key_variable)}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": text}
        ],
        # "provider": {
        #     "order": [
        #         "Fireworks"
        #     ],
        #     "allow_fallbacks": False
        # },
        **kwargs
    }
    response = await fetch(logger, session, base_url, headers, body, timeout=timeout)
    logprob = None
    logprob_token = None
    if response is not None:
        try:
            # logprob = response['choices'][0]['logprobs']['content']
            # logprob_token = logprob[0] if logprob else None
            # logprob = logprob[0]['logprob'] if logprob else None
            response = response["choices"][0]['message']
            if "reasoning_content" in response:
                body["reasoning_content"] = response["reasoning_content"]
            elif "reasoning" in response:
                body["reasoning"] = response["reasoning"]
            response = response["content"]
        except KeyError:
            logger.error(f"Response is badly formatted:\n{response}\n{body}")
            logger.error(traceback.format_exc())
            response = ""
    return response, body, logprob, logprob_token


async def process_dataset(dataset: List[Dict[str, Any]], model_id: str, system_prompt: str, user_wrap: str,
                          max_tokens: int, temperature: float, max_concurrent_requests: int, timeout: int, base_url: str, api_key_variable: str,
                          logger, save_file: Optional[Path] = None) -> List[Dict[str, Any]]:
    results = list()
    lock = asyncio.Lock()
    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(max_concurrent_requests)  # Limit the number of concurrent requests

        async def sem_task(example):
            text = user_wrap.format(preprocessing(example["text"]))
            async with sem:
                # result, body, logprob, logprob_token = await process_request(logger, model_id, system_prompt, text,
                #                                                              session, timeout, base_url, api_key_variable,
                #                                                              max_tokens=max_tokens, logprobs=True, temperature=temperature)

                result, body, logprob, logprob_token = await process_request(logger, model_id, system_prompt, text,
                                                                             session, timeout, base_url, api_key_variable,
                                                                             max_tokens=max_tokens, temperature=temperature,
                                                                             do_sample=temperature > 0, num_beams=2)
                out = {
                    "id": example["id"],
                    "output": result,
                    "input_text": example["text"],
                    "confidence": np.exp(logprob) if logprob is not None else None,
                    "confidence_token": logprob_token,
                    "label": example["label"],
                    "model_input": body,
                    "example": example
                }
                results.append(out)
                if save_file is not None:
                    await save_results_to_disk(save_file, results, lock)

        tasks = [sem_task(ex) for ex in dataset]
        await tqdm_asyncio.gather(*tasks)
    return results


async def main(model_id: str, dataset: List[Dict[str, Any]], max_tokens: int, temperature: float,
               results_dir: str, prompt: str, max_concurrent_requests: int, timeout: int, base_url: str, api_key_variable: str):
    outdir = Path(results_dir) / test_set_name / Path(model_id).name / Path(prompt).with_suffix("").name

    output_file = outdir / "output.json"
    tmp_output_file = outdir / "temp_output.json"  # temp file where to save progress in case of apocalypse

    log_path = outdir
    log_path.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_path, LOG_NAME)
    logger.debug(f"Set up logger in dir '{str(log_path)}'.")

    evaluate_rationale = prompt.endswith("p3.txt") or prompt.endswith("p5.txt")
    prompt_file = Path(prompt)
    system_prompt: str = prompt_file.read_text()
    user_wrap: str = Path(str(prompt_file.with_suffix("")) + "-wrap.txt").read_text()

    evaluator = Evaluator(logger, outdir)

    if not args.force and output_file.exists():
        logger.info(f"**** Evaluate existing results in '{str(output_file)}'")
        results = load_json(output_file)
        existing_ids = {e["id"] for e in results}
        dataset_ids = {e["id"] for e in dataset}
        existing_ids_in_dataset = existing_ids & dataset_ids

        # check the text of existing ids and if they are changed, and if yes, they should be re-evaluated
        results_by_id = {e["id"]: e for e in results}
        for e in dataset:
            example_id = e["id"]
            if example_id in existing_ids_in_dataset:
                res = results_by_id[example_id]
                if e["text"] != res["input_text"]:
                    # should be evaluated
                    existing_ids_in_dataset.remove(example_id)

        ids_to_re_evaluate = dataset_ids - existing_ids_in_dataset

        if ids_to_re_evaluate:
            ## TMP debug
            logger.info(f"**** Re evaluation of {len(ids_to_re_evaluate)} existing results")
            save_json(outdir / "reval.json", list(ids_to_re_evaluate))

            # Evaluate only different ids
            examples_to_re_evaluate = [e for e in dataset if e["id"] in ids_to_re_evaluate]
            # call API only with selected ids
            results_re_evaluated = await process_dataset(examples_to_re_evaluate, model_id, system_prompt, user_wrap,
                                                         max_tokens, temperature, max_concurrent_requests, timeout, base_url, api_key_variable, logger, tmp_output_file)
            results_from_dataset = [r for r in results if r["id"] in existing_ids_in_dataset]
            results_new = results_from_dataset + results_re_evaluated
            if len(results_new) != len(results):
                shutil.move(output_file, output_file.with_suffix(".bak.json"))
            logger.info(
                f"Existing {len(existing_ids)} results. Re-evaluated {len(results_re_evaluated)} "
                f"results. Kept {len(results_from_dataset)} results from existing. "
                f"Total is {len(results_new)} results.")
            # overwrite the new with the old results
            results = results_new
            save_json(output_file, results)
            os.remove(tmp_output_file)
        else:
            logger.info("**** Nothing to re-evaluate")
    else:
        logger.info(f"**** Going from scratch")
        outdir.mkdir(exist_ok=True, parents=True)
        results = await process_dataset(dataset, model_id, system_prompt, user_wrap,
                                        max_tokens, temperature, max_concurrent_requests, timeout, base_url, api_key_variable, logger, tmp_output_file)
        # Save results to a JSON file
        save_json(output_file, results)
        os.remove(tmp_output_file)

    # removed biased
    # import re
    # bias_pattern = re.compile(r"old.*never die.*they|\WTom\W|was only.*daughter.*but|doctor.*doctor|used to.*but|^when the", flags=re.IGNORECASE)
    # non_biased_ids = {e["id"] for e in dataset if not bias_pattern.search(e["text"])}
    # b_len = len(dataset) - len(non_biased_ids)
    # print(f"Removed {b_len}")
    # # non_biased_ids = random.sample(non_biased_ids, b_len)
    # dataset = [e for e in dataset if e["id"] in non_biased_ids]
    # results = [e for e in results if e["id"] in non_biased_ids]

    metrics = evaluator.evaluate_metrics(dataset, results, evaluate_rationale, evaluate_rationale)
    metrics_0 = evaluator.evaluate_metrics(dataset, results, False, evaluate_rationale, pos_class=0, hom_het_evaluation=False, re_parse_results=False)
    metrics["precision_0"] = metrics_0["precision"]
    metrics["recall_0"] = metrics_0["recall"]

    # save_json(outdir / "metrics_unbiased.json", metrics)
    save_json(outdir / "metrics.json", metrics)
    desired_order = ['accuracy', 'f1', 'precision', 'recall']
    if evaluate_rationale:
        desired_order.extend(
            ['het_recall', 'hom_recall', 'het_support', 'hom_support', 'kw_agreement_2', 'kw_agreement_1',
             'kw_agreement_0', 'kw_agreement'])
    df_metrics = pd.Series(metrics).reindex(desired_order)
    df_metrics.to_csv(outdir / "metrics.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="OpenRouter model ID")
    parser.add_argument("--prompt", type=str, required=True, help="Path to the prompt txt file")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the test data JSON file")
    parser.add_argument("--outdir", type=str, default="results/llm", help="Directory to save results")
    parser.add_argument("--max_tokens", type=int, default=128, help="Maximum tokens for the response")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature for the response")
    parser.add_argument("--concurrence", type=int, default=2, help="Max concurrent requests")
    parser.add_argument("--timeout", type=int, default=30, help="Max timeout for each requests in seconds")
    parser.add_argument("--provider", type=str, choices=['openrouter', 'deepseek'], required=True, help="Provider of choice. Will set the key and base url")
    parser.add_argument("--position", type=str, choices=['after', 'before'], required=True,
                        help="Ask rationale 'before' or 'after' prompt")
    parser.add_argument("--force", action="store_true",
                        help="Force results over-write, and disable re-evaluation (it won't keep anything existing)")
    args = parser.parse_args()

    if args.provider == "deepseek":
        _base_url = "https://api.deepseek.com/chat/completions"
        _api_key = "DEEPSEEK_API_KEY"
    elif args.provider == "openrouter":
        _base_url = "https://openrouter.ai/api/v1/chat/completions"
        _api_key = "OPENROUTER_API_KEY"
    else:
        raise ValueError(f"Unsupported provider {args.provider}. Must be one of ['openrouter', 'deepseek']")

    # Load dataset
    dataset = load_json(Path(args.test_data))
    random.shuffle(dataset)
    # dataset = [d for d in dataset if d["id"] in ["pos_12"]]
    test_set_name = Path(args.test_data).with_suffix("").name

    # Run the main function
    asyncio.run(main(args.model_id, dataset, args.max_tokens, args.temp, args.outdir, args.prompt, args.concurrence, args.timeout, _base_url, _api_key))
