from functools import partial
from pathlib import Path
from typing import Tuple

from datasets import Split, load_from_disk
from transformers import TrainingArguments, set_seed, EarlyStoppingCallback

from finetuning.llama.utils.data import format_chat_dataset, apply_template
from utils.io import load_json
from utils.log import setup_logging


# from trl import SFTTrainer
# from unsloth import FastLanguageModel, is_bfloat16_supported
# from unsloth.chat_templates import get_chat_template


def train(config) -> Tuple:
    set_seed(config.seed)

    train_dataset_path = Path(config.train_data)
    train_examples = load_json(train_dataset_path)
    train_dataset_name = train_dataset_path.with_suffix("").name

    outdir = Path(config.outdir) / train_dataset_name / config.model_name / Path(config.prompt).name
    train_dump_data_dir = (train_dataset_path.parent / train_dataset_name).with_suffix(".hf")

    # Logging
    log_path = outdir
    log_path.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_path, "ft-llama")
    logger.debug(f"Set up logger in dir '{str(log_path)}'.")

    # System/User prompts
    add_rationale = config.prompt.endswith("p3.txt") or config.prompt.endswith("p5.txt")
    extended_rationale = config.prompt.endswith("p5.txt")
    system_prompt: str = Path(config.prompt).read_text()
    user_wrap: str = Path(str(Path(config.prompt).with_suffix("")) + "-wrap.txt").read_text()

    # Format the data as chats and save them to disk
    if not train_dump_data_dir.exists():
        format_chat_dataset(train_examples, add_rationale, extended_rationale, system_prompt, user_wrap,
                            train_dump_data_dir, Split.TRAIN)

    training_set = load_from_disk(str(train_dump_data_dir))
    logger.info("------------ Training set length: {} ------------ ".format(len(training_set)))

    if config.val_data:
        val_dataset_path = Path(config.val_data)
        val_examples = load_json(val_dataset_path)
        val_dataset_name = val_dataset_path.with_suffix("").name
        val_dump_data_dir = (val_dataset_path.parent / val_dataset_name).with_suffix(".hf")
        logger.info(f"Set up validation dataset in dir '{str(val_dump_data_dir)}'.")
        if not val_dump_data_dir.exists():
            format_chat_dataset(val_examples, add_rationale, extended_rationale, system_prompt, user_wrap,
                                val_dump_data_dir,
                                Split.VALIDATION)
        validation_set = load_from_disk(str(val_dump_data_dir))
        logger.info("------------ Validation set length: {} ------------ ".format(len(validation_set)))
    else:
        validation_set, val_examples = None, None
        logger.info(f"No validation dataset available. Disabling Early Stopping and validation metrics.")

    # Init UNSLOTH model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_id,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
        dtype=None
    )

    # Use PEFT
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=config.seed,
        max_seq_length=config.max_seq_length,
        use_rslora=False  # We support rank stabilized LoRA
    )

    tokenizer = get_chat_template(
        tokenizer,
        mapping={"role": "from", "content": "value", "user": "user", "assistant": "assistant", "system": "system"},
        chat_template="chatml",
    )

    # Apply formatting to both data
    apply_template_with_tokenizer = partial(apply_template, tokenizer)
    training_set = training_set.map(apply_template_with_tokenizer, batched=True)

    # Disabled if we do not use custom metrics
    # bos_token = "<|im_start|>"
    # compute_metrics_fn = partial(compute_metrics, model, tokenizer, train_examples, system_prompt, user_wrap,
    #                              False, args, logger, outdir, bos_token, args.temp)

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config.patience,  # Number of evaluations with no improvement before stopping
        early_stopping_threshold=0.01,  # Minimum change to qualify as an improvement
    )

    if validation_set is not None:
        # Add validation-related arguments
        validation_set = validation_set.map(apply_template_with_tokenizer, batched=True)

        trainer_args = dict(
            model=model,
            tokenizer=tokenizer,
            train_dataset=training_set,
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
            dataset_num_proc=2,
            packing=config.packing,
            args=TrainingArguments(
                learning_rate=config.lr,
                lr_scheduler_type="linear",
                per_device_train_batch_size=config.bs,
                gradient_accumulation_steps=config.ga,
                num_train_epochs=config.epochs,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=config.wd,
                warmup_steps=10,
                output_dir=outdir,
                seed=config.seed,
                do_train=True,
                eval_strategy="steps",
                save_strategy="steps",
                do_eval=True,
                save_total_limit=1,
                eval_steps=10,
                save_steps=10,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            ),
            eval_dataset=validation_set,
            callbacks=[early_stopping_callback]
        )
    else:
        # Only training-related arguments
        trainer_args = dict(
            model=model,
            tokenizer=tokenizer,
            train_dataset=training_set,
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
            dataset_num_proc=2,
            packing=config.packing,
            args=TrainingArguments(
                learning_rate=config.lr,
                lr_scheduler_type="linear",
                per_device_train_batch_size=config.bs,
                gradient_accumulation_steps=config.ga,
                num_train_epochs=config.epochs,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=config.wd,
                warmup_steps=10,
                output_dir=outdir,
                seed=config.seed,
                do_train=True,
                eval_strategy="no",
                save_strategy="no",
                do_eval=False,
            ))

    trainer = SFTTrainer(**trainer_args)
    trainer.train()

    model.save_pretrained_merged(str(outdir / "model"), tokenizer, save_method="lora")
    # model.save_pretrained_merged(str(outdir / "model"), tokenizer, save_method="merged_16bit")
    return model, tokenizer, train_examples, val_examples, logger, outdir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Unsloth model ID")
    parser.add_argument("--model_name", type=str, required=True, help="Name to identify the model")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt type. Must be one of the names in 'prompts/new'")
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
    parser.add_argument("--ga", type=int, default=2, help="gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3.0e-4, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.01, help="adam weight decay")
    parser.add_argument("--packing", action="store_true", help="use packing to combine examples")
    parser.add_argument("--lora_r", type=int, help="lora rank, the higher the more model will learn")
    parser.add_argument("--lora_alpha", type=int, help="lora alpha, typically equal to r")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature for the response")
    # parser.add_argument("--logdir", type=str, default="logs/llm")
    # parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Load dataset
    train(args)
