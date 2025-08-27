#!/bin/bash

RUNPATH=/home/alessandro/work/repo/task-testbed
cd $RUNPATH || exit
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$RUNPATH

# WHERE TO SAVE ALL PLOTS
PLOT_FOLDER=results/plots
mkdir -p $PLOT_FOLDER

# barplot_unbiased_vs_puneval_dataset.png
# boxplot_f1_distribution_nap_+_joker_dataset.png
# boxplot_f1_distribution_unbiased_dataset.png
# violin_precision_prompt_distribution_biased_subsets.png
# violin_recall_prompt_distribution_biased_subsets.png
RESULTS_BASE_FOLDER=results/standard_prompt/base_data
python scripts/generate_plots/prompting_metrics_visualization.py \
  $RESULTS_BASE_FOLDER/run_1 \
  $RESULTS_BASE_FOLDER/run_2 \
  $RESULTS_BASE_FOLDER/run_3
mv $RESULTS_BASE_FOLDER/create_sheet_output/*.png $PLOT_FOLDER/

RESULTS_BASE_FOLDER=results/standard_prompt/substitution_data
# barplot_best_recall_by_substitution_best_prompt.png
# violin_ALL_rationale_effect_recall_by_prompt.png
python -O scripts/generate_plots/substitution_experiment_visualization.py \
    $RESULTS_BASE_FOLDER/run_1 \
    $RESULTS_BASE_FOLDER/run_2 \
    $RESULTS_BASE_FOLDER/run_3 \
    --metric accuracy

# barplot_best_precision_by_substitution_best_prompt.png
python -O scripts/generate_plots/substitution_experiment_visualization.py \
    $RESULTS_BASE_FOLDER/run_1 \
    $RESULTS_BASE_FOLDER/run_2 \
    $RESULTS_BASE_FOLDER/run_3 \
    --metric precision
mv $RESULTS_BASE_FOLDER/create_sheet_output/*.png $PLOT_FOLDER/
rm $PLOT_FOLDER/violin_ALL_rationale_effect_precision_by_prompt.png # we do not need this


# boxplot_c_correlation_confidence_and_prediction_class.png
python scripts/generate_plots/plot_confidence_correlation.py \
  results/debug/confidence/substitution_data/run_1/all \
  --output_dir $PLOT_FOLDER/

# test_tsne_separation_no_ft.png
# nap_joker_tsne_separation_no_ft.png
python scripts/generate_plots/plot_embeddings_separation.py \
  --model_dir FacebookAI/roberta-large \
  --test_files data/legacy/nap_joker.json data/public/puneval/test.json \
  --output_dir $PLOT_FOLDER/ \
  --seed 73 \
  --suffix _no_ft

# test_tsne_separation.png
python scripts/generate_plots/plot_embeddings_separation.py \
  --model_dir dumps/roberta-large \
  --test_files data/public/puneval/test.json \
  --output_dir $PLOT_FOLDER/ \
  --seed 73

# error_analysis_categories_per_model_nonexclusive.png
python gform/error_analysis.py gform/study/results \
  --output_folder $PLOT_FOLDER/
