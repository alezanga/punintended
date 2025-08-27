# Pun Unintended: LLMs and the Illusion of Humor Understanding

Code and data for paper `Pun Unintended: LLMs and the Illusion of Humor Understanding`, accepted at EMNLP 2025.

## Info

Preprint: TODO

Released data is described in the specific [readme file](./data/public/readme.md).

## Usage

Run experiments with bash scripts in `scripts/run_experiments`

- `tests-oai`: OpenAI GPT-4o tests (OpenAI API)
- `tests-ds`: DeepSeek R1 tests (DeepSeek API)
- `tests-or`: all other LLMs (OpenRouter API)
- `train_test_vanilla_llama.sh` launch training and metrics evaluation of Llama3.1-8B

Fine-tune RoBERTa model:

```shell
python finetuning/encoder/train_and_test_script.py \
    --mode train \
    --model_name_or_path FacebookAI/roberta-large \
    --model_dir dumps/roberta-large \
    --train_file data/public/puneval/train.json \
    --val_file data/public/puneval/val.json \
    --max_epochs 4 \
    --batch_size 32 \
    --learning_rate 1.5e-5 \
    --test_files data/public/puneval/test.json data/public/nap.json data/private/joker/ruined.json \
    --output_dir results/roberta-large
```

Reproduce paper plots using `scripts/generate_plots/generate_all_plots.sh`.
Results are reported in CSV files placed inside folder: `$RESULTS_BASE_FOLDER/create_sheet_output/`. The base folder can
be set inside the script.

## Code organization

```
.
├── data
│   └── public              # public datasets
├── dataset_generation      # analyze datasets, generate new examples
├── evaluation              # predictions and rationales evaluation
├── finetuning              # train encoder (RoBERTa) and Llama (unused)
├── prompts                 # .txt files contianing the prompts used
├── results                 # folder will be created for plots/results
├── scripts                 # run experiments and generate plots/results
├── secrets                 # tokens for GForm and other APIs
└── utils                   # code utilities
```

Inside `prompts` folder, the `px-wrap.txt` files contain the text used in the _user_ prompt, while the `px.txt` files
contain _system_ prompts.