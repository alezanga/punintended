#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### python finetuning/encoder/bert.py --mode train --model_name_or_path FacebookAI/roberta-large --model_dir dumps/roberta-large --train_file data/puneval/train_puns_nodup.json --val_file data/puneval/val_puns.json --max_epochs 20 --batch_size 32 --learning_rate 2e-5 --test_files data/puneval/test_puns.json data/new/new.json data/experiment/all.json --output_dir results/roberta-large

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed, EarlyStoppingCallback,
)

from utils.io import load_json, save_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Set seed for reproducibility
set_seed(42)


# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train or evaluate a RoBERTa model for binary text classification."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval"],
        help="Run mode: 'train' or 'eval'.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="roberta-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory to save model checkpoints (in train mode) or load from (in eval mode).",
    )

    # Training specific arguments
    parser.add_argument(
        "--train_file",
        type=str,
        help="Path to the training JSON file (required in train mode).",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        help="Path to the validation JSON file (required in train mode).",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=15,
        help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size per device during training and evaluation."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate for AdamW optimizer."
    )

    # Evaluation specific arguments
    parser.add_argument(
        "--test_files",
        nargs="+",  # Allows one or more test files
        type=str,
        help="Path(s) to the test JSON file(s) (required in eval mode).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save evaluation metrics JSON files (required in eval mode).",
    )

    # Common arguments
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization."
    )

    args = parser.parse_args()

    # --- Validate Arguments ---
    if args.mode == "train":
        if not args.train_file or not args.val_file:
            parser.error("--train_file and --val_file are required in train mode.")
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        logging.info(f"Running in TRAIN mode.")
    elif args.mode == "eval":
        if not args.test_files or not args.output_dir:
            parser.error("--test_files and --output_dir are required in eval mode.")
        if not os.path.isdir(args.model_dir):
            parser.error(f"--model_dir '{args.model_dir}' does not exist or is not a directory in eval mode.")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logging.info(f"Running in EVAL mode.")

    set_seed(args.seed)
    return args


# --- Data Processing ---
def preprocess_data(examples: Dict[str, List], tokenizer: AutoTokenizer, max_len: int) -> Dict:
    """Tokenizes the input text."""
    # Ensure 'text' field exists and handle potential None values if necessary
    texts = [str(t) if t is not None else "" for t in examples["text"]]
    result = tokenizer(
        texts,
        padding="max_length",  # Pad to max_len during preprocessing
        truncation=True,
        max_length=max_len,
        return_tensors=None,  # Return lists to be handled by Dataset
    )
    # Rename 'label' column to 'labels' for the trainer
    # if "label" in examples:
    #     result["labels"] = examples["label"]
    return result


# --- Metrics Computation ---
def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    """Computes metrics for evaluation."""
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    labels = p.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'  # Use 'binary' for binary classification
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# --- Main Function ---
def main():
    args = parse_arguments()

    # --- Load Tokenizer ---
    logging.info(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.mode == "train":
        # --- Load Data ---
        logging.info("Loading training and validation data...")
        train_data = load_json(Path(args.train_file))
        val_data = load_json(Path(args.val_file))

        # --- Convert to Hugging Face Dataset ---
        # Assumes load_json returns List[Dict]
        # Convert List[Dict] to Dict[str, List] format for Dataset.from_dict
        train_dict = {key: [dic[key] for dic in train_data] for key in train_data[0] if key in ["text", "label", "id"]}
        val_dict = {key: [dic[key] for dic in val_data] for key in val_data[0] if key in ["text", "label", "id"]}

        raw_datasets = DatasetDict({
            "train": Dataset.from_dict(train_dict),
            "validation": Dataset.from_dict(val_dict),
        })
        logging.info(f"Raw data loaded: {raw_datasets}")

        # --- Tokenize Data ---
        logging.info("Tokenizing data...")
        tokenized_datasets = raw_datasets.map(
            lambda examples: preprocess_data(examples, tokenizer, args.max_seq_length),
            batched=True,
            remove_columns=[col for col in raw_datasets["train"].column_names if col not in ["input_ids", "attention_mask", "label"]],  # Keep necessary columns
            desc="Running tokenizer on dataset",
        )
        # Ensure 'labels' column exists after tokenization
        if "label" in tokenized_datasets["train"].column_names:
            tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        logging.info(f"Tokenized data prepared: {tokenized_datasets}")

        # --- Load Model ---
        logging.info(f"Loading model for sequence classification: {args.model_name_or_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=2  # Binary classification
        )

        # --- Data Collator ---
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # --- Training Arguments ---
        logging.info("Configuring training arguments...")
        training_args = TrainingArguments(
            output_dir=args.model_dir,
            num_train_epochs=args.max_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            evaluation_strategy="epoch",  # Evaluate at the end of each epoch
            save_strategy="epoch",  # Save at the end of each epoch
            logging_strategy="epoch",  # Log at the end of each epoch
            load_best_model_at_end=True,  # Load the best model based on metric
            metric_for_best_model="f1",  # Use F1 score for early stopping
            greater_is_better=True,  # Higher F1 is better
            save_total_limit=1,  # Keep only the last checkpoint (best model will be saved at the end)
            fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA is available
            report_to="none",  # Disable external reporting (e.g., wandb)
            seed=args.seed,
        )

        # --- Initialize Trainer ---
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Add this line
        )

        # --- Train ---
        logging.info("Starting training...")
        train_result = trainer.train()
        logging.info("Training finished.")

        # --- Save final best model, tokenizer, and training state ---
        # The best model is already loaded due to load_best_model_at_end=True
        logging.info(f"Saving the best model to {args.model_dir}")
        trainer.save_model(args.model_dir)  # Saves model and tokenizer
        trainer.save_state()  # Saves training state

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # --- Evaluate on Validation Set (Optional, but good practice) ---
        logging.info("Evaluating the best model on the validation set...")
        eval_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
        logging.info(f"Validation metrics: {eval_metrics}")
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)  # Saves to args.model_dir/all_results.json


    elif args.mode == "eval":
        # --- Load Model and Tokenizer from Checkpoint ---
        logging.info(f"Loading model and tokenizer from {args.model_dir}")
        # Check if model_dir exists (already done in arg parsing, but double-check)
        if not os.path.isdir(args.model_dir):
            logging.error(f"Model directory not found: {args.model_dir}")
            sys.exit(1)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
            # Tokenizer is usually saved alongside the model
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        except Exception as e:
            logging.error(f"Error loading model or tokenizer from {args.model_dir}: {e}")
            sys.exit(1)

        # --- Data Collator ---
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # --- Initialize Trainer for Evaluation ---
        # No need for full TrainingArguments, just output_dir and batch size
        eval_args = TrainingArguments(
            output_dir=args.output_dir,  # Temporary dir for predictions if needed
            per_device_eval_batch_size=args.batch_size,
            fp16=torch.cuda.is_available(),
            report_to="none",
            seed=args.seed,
        )

        trainer = Trainer(
            model=model,
            args=eval_args,  # Use minimal args for evaluation
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # --- Process each Test File ---
        for test_file_path in args.test_files:
            logging.info(f"Processing test file: {test_file_path}")

            # --- Load Test Data ---
            test_data = load_json(Path(test_file_path))
            if not test_data:
                logging.warning(f"No data loaded from {test_file_path}, skipping.")
                continue

            # Convert to Hugging Face Dataset
            test_dict = {key: [dic[key] for dic in test_data] for key in test_data[0] if key in ["text", "label", "id"]}
            raw_test_dataset = Dataset.from_dict(test_dict)

            # --- Tokenize Test Data ---
            logging.info(f"Tokenizing test data from {test_file_path}...")
            tokenized_test_dataset = raw_test_dataset.map(
                lambda examples: preprocess_data(examples, tokenizer, args.max_seq_length),
                batched=True,
                remove_columns=[col for col in raw_test_dataset.column_names if col not in ["input_ids", "attention_mask", "label"]],
                desc=f"Running tokenizer on {os.path.basename(test_file_path)}",
            )
            # Ensure 'labels' column exists
            if "label" in tokenized_test_dataset.column_names:
                tokenized_test_dataset = tokenized_test_dataset.rename_column("label", "labels")
            elif "labels" not in tokenized_test_dataset.column_names:
                logging.error(f"Test file {test_file_path} must contain a 'label' column for evaluation.")
                continue  # Skip this file

            # --- Evaluate ---
            logging.info(f"Evaluating model on {os.path.basename(test_file_path)}...")
            results = trainer.predict(tokenized_test_dataset)
            metrics = results.metrics  # This contains {'test_loss': ..., 'test_accuracy': ..., 'test_f1': ..., etc.}

            # --- Save Metrics ---
            dataset_name = Path(test_file_path).stem  # Get filename without extension
            output_metrics_file = Path(args.output_dir, f"{dataset_name}_metrics.json")

            # Clean up metric keys (remove 'test_' prefix if present)
            cleaned_metrics = {k.replace('test_', ''): v for k, v in metrics.items()}

            logging.info(f"Saving metrics for {dataset_name} to {output_metrics_file}")
            logging.info(f"Metrics: {cleaned_metrics}")
            output_metrics_file.parent.mkdir(parents=True, exist_ok=True)
            save_json(output_metrics_file, cleaned_metrics)

    logging.info("Script finished.")


if __name__ == "__main__":
    main()
