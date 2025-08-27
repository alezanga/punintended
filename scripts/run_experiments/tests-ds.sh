#!/bin/bash

RUNPATH=/home/alessandro/work/repo/task-testbed
cd $RUNPATH || exit
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$RUNPATH

BASE_DATA_FILES=("data/public/punny_pattern/daughter.json"
                 "data/public/punny_pattern/doctor.json"
                 "data/public/punny_pattern/never_die.json"
                 "data/public/punny_pattern/tom.json"
                 "data/public/punny_pattern/used.json"
                 "data/public/punny_pattern/when.json"
                 "data/public/nap.json"
                 "data/public/puneval/test.json"
                 "data/private/joker/ruined.json")
SUBSTITUTION_DATA_FILES=("data/public/pun_break.json")
PROMPTS=("prompts/p1.txt" "prompts/p2.txt" "prompts/p3.txt" "prompts/p5.txt")
MODEL_IDS=("deepseek-reasoner")

function run_experiments {
  local base_outdir="$1" # Base output directory
  local test_data_files=("${!2}") # Array of test data files passed by reference

  for EVAL_MODEL_ID in "${MODEL_IDS[@]}"; do
    for TEST_DATA in "${test_data_files[@]}"; do
      for PROMPT in "${PROMPTS[@]}"; do
        python scripts/run_experiments/main_or.py \
            --model_id "$EVAL_MODEL_ID" \
            --prompt "$PROMPT" \
            --test_data "$TEST_DATA" \
            --outdir "${base_outdir}/run_1" \
            --max_tokens 1024 \
            --temp 0.0 \
            --concurrence 3 \
            --position after \
            --provider deepseek \
            --timeout 900

        python scripts/run_experiments/main_or.py \
            --model_id "$EVAL_MODEL_ID" \
            --prompt "$PROMPT" \
            --test_data "$TEST_DATA" \
            --outdir "${base_outdir}/run_2" \
            --max_tokens 1024 \
            --temp 0.0 \
            --concurrence 3 \
            --position after \
            --provider deepseek \
            --timeout 900

        python scripts/run_experiments/main_or.py \
            --model_id "$EVAL_MODEL_ID" \
            --prompt "$PROMPT" \
            --test_data "$TEST_DATA" \
            --outdir "${base_outdir}/run_3" \
            --max_tokens 1024 \
            --temp 0.0 \
            --concurrence 3 \
            --position after \
            --provider deepseek \
            --timeout 900
      done
    done
  done
}


# RUN
run_experiments "results/standard_prompt/base_data" BASE_DATA_FILES[@]
run_experiments "results/standard_prompt/substitution_data" SUBSTITUTION_DATA_FILES[@]