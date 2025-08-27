from pathlib import Path

from finetuning.llama.scripts.inference import infer
from finetuning.llama.scripts.train import train
from utils.io import load_json
from utils.log import setup_logging

DATASETS = [
    "data/public/punny_pattern/daughter.json",
    "data/public/punny_pattern/doctor.json",
    "data/public/punny_pattern/never_die.json",
    "data/public/punny_pattern/tom.json",
    "data/public/punny_pattern/used.json",
    "data/public/punny_pattern/when.json",
    "data/public/nap.json",
    "data/public/puneval/test.json",
    "data/public/pun_break.json"
]

PROMPTS = [
    "prompts/p1.txt",
    "prompts/p2.txt",
    "prompts/p3.txt",
    "prompts/p5.txt"
]


def train_and_test(config) -> None:
    # TRAIN
    if not config.reval:
        model, tokenizer, _, val_examples, _, _ = train(config)
    else:
        # trick is to set the model to something that is NOT None
        model, tokenizer, val_examples = 1, None, None
    train_prompt = config.prompt.strip()

    # TEST
    for test_prompt in PROMPTS:
        # test all prompts in testing
        config.prompt = test_prompt.strip()
        if test_prompt.strip() != train_prompt:
            # Only test training = testing prompt
            continue
        if test_prompt.strip() != train_prompt:
            prompt_identifier = Path(train_prompt).with_suffix("").name + "_" + Path(test_prompt).name
        else:
            prompt_identifier = Path(config.prompt).name

        for dataset_path in DATASETS:
            dataset_path = Path(dataset_path)
            test_examples = load_json(dataset_path)
            dataset_name = dataset_path.with_suffix("").name
            outdir = Path(config.result_dir) / dataset_name / config.model_name / prompt_identifier

            if config.reval:
                if not outdir.exists():
                    # Results not present
                    print(f"Skipping re-eval of folder '{outdir}'")
                    continue
                else:
                    print(f"Re-evaluating folder '{outdir}'")

            log_path = outdir
            log_path.mkdir(parents=True, exist_ok=True)
            logger = setup_logging(log_path, "infer-ft-llama")
            logger.debug(f"Set up logger in dir '{str(log_path)}'.")

            if not config.reval and (outdir / "metrics.json").exists():
                logger.error(f"*** Metric file '{str(outdir)}' already exists, skipping to avoid overwriting results.")
                continue

            infer(config, model, tokenizer, test_examples, logger, outdir)

        if val_examples:
            # validation test
            dataset_name = "validation"
            outdir = Path(config.result_dir) / dataset_name / config.model_name / prompt_identifier

            log_path = outdir
            log_path.mkdir(parents=True, exist_ok=True)
            logger = setup_logging(log_path, "infer-ft-llama")
            logger.debug(f"Set up logger in dir '{str(log_path)}'.")

            infer(config, model, tokenizer, val_examples, logger, outdir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Unsloth model ID")
    parser.add_argument("--model_name", type=str, required=True, help="Name to identify the model")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt type during training. Must be one of the names in 'prompts/new'")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the train data JSON file")
    parser.add_argument("--val_data", type=str, required=False, default=None,
                        help="Path to the validation data JSON file. Do not set to disable validation and early stopping.")
    parser.add_argument("--outdir", type=str, default="dumps/llm", help="Directory to save model and training dumps")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max new tokens in the answer")
    parser.add_argument("--epochs", type=float, default=1, help="training epochs")
    parser.add_argument("--patience", type=int, default=1, help="patience when doing ES")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--bs", type=int, default=8, help="per GPU batch size")
    parser.add_argument("--ga", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3.0e-4, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.01, help="adam weight decay")
    parser.add_argument("--packing", action="store_true", help="use packing to combine examples")
    parser.add_argument("--lora_r", type=int, help="lora rank, the higher the more model will learn")
    parser.add_argument("--lora_alpha", type=int, help="lora alpha, typically equal to r")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature for the response")
    parser.add_argument("--boa", type=str, default="<|im_start|>assistant",
                        help="String to look for in decoded prediction that marks the start of answer")
    parser.add_argument("--reval", action="store_true", help="re-evaluate existing results only")
    parser.add_argument("--result_dir", type=str, required=True, help="Where to save results and metrics (directory)")

    # parser.add_argument("--logdir", type=str, default="logs/llm")
    # parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    train_and_test(args)
