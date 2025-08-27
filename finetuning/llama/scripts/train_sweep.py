import wandb

from finetuning.llama.scripts.inference import infer
from finetuning.llama.scripts.train import train


def sweep():
    wandb.init()
    config = wandb.config

    model, tokenizer, train_examples, val_examples, logger, outdir = train(config)

    metrics = infer(config, model, tokenizer, val_examples, logger, outdir)
    metrics = {f"eval_{k}": v for k, v in metrics.items()}
    wandb.log(metrics)
    wandb.finish()


if __name__ == "__main__":
    sweep()
