from pathlib import Path
from typing import Dict

import pandas as pd
import transformers
from transformers import set_seed

from evaluation.evaluator import Evaluator
from finetuning.llama.utils.evaluation import get_model_predictions
from utils.io import load_json, save_json
from utils.log import setup_logging


def infer(config, model=None, tokenizer=None, examples=None, logger=None, outdir: Path = None) -> dict:
    set_seed(config.seed)

    add_rationale = config.prompt.endswith("p3.txt") or config.prompt.endswith("p5.txt")
    system_prompt: str = Path(config.prompt).read_text()
    user_wrap: str = Path(str(Path(config.prompt).with_suffix("")) + "-wrap.txt").read_text()

    if model is None:
        dataset_path = Path(config.test_data)
        examples = load_json(dataset_path)
        dataset_name = dataset_path.with_suffix("").name

        outdir = Path(config.outdir) / dataset_name / config.model_name / Path(config.prompt).name

        # Logging
        log_path = outdir
        log_path.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(log_path, "infer-ft-llama")
        logger.debug(f"Set up logger in dir '{str(log_path)}'.")

        if (not hasattr(config, "reval") or not config.reval) and (outdir / "metrics.json").exists():
            logger.error(f"*** Metric file '{str(outdir)}' already exists, exiting to avoid overwriting results.")
            return dict()

        # initialize the model
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_id, trust_remote_code=True,
                                                                  device_map="auto", load_in_4bit=config.quant4)
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_id)
    else:
        logger.info("Inference after sweep")

    metrics = evaluate(model, tokenizer, examples, system_prompt, user_wrap, add_rationale, config, logger, outdir)
    return metrics


def evaluate(model, tokenizer, examples, system_prompt, user_wrap, add_rationale, config, logger, outdir) \
        -> Dict[str, float]:
    output_file = outdir / "output.json"
    if not hasattr(config, "reval") or not config.reval:
        model.eval()
        outputs = get_model_predictions(model, tokenizer, examples, system_prompt, user_wrap, config, logger)
        save_json(output_file, outputs)
    else:
        # re-evaluate only
        outputs = load_json(output_file)

    evaluator = Evaluator(logger, outdir)
    metrics = evaluator.evaluate_metrics(examples, outputs, False, add_rationale)
    save_json(outdir / "metrics.json", metrics)
    desired_order = ['accuracy', 'f1', 'precision', 'recall']
    if add_rationale:
        desired_order.extend(
            ['het_recall', 'hom_recall', 'het_support', 'hom_support', 'kw_agreement_2', 'kw_agreement_1',
             'kw_agreement_0', 'kw_agreement'])
    df_metrics = pd.Series(metrics).reindex(desired_order)
    df_metrics.to_csv(outdir / "metrics.csv")
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Unsloth model ID")
    parser.add_argument("--model_name", type=str, required=True, help="Name to identify the model")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt type. Must be a txt file")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the test data JSON file")
    parser.add_argument("--outdir", type=str, default="dumps/llm", help="Directory to save model and training dumps")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max new tokens in the answer")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature for the response")
    parser.add_argument("--boa", type=str, default="<|im_start|>assistant",
                        help="String to look for in decoded prediction that marks the start of answer")
    parser.add_argument("--quant4", action="store_true", help="Whether to use 4bit quantization")
    parser.add_argument("--reval", action="store_true", help="re-evaluate existing results only")
    args = parser.parse_args()

    infer(args)
