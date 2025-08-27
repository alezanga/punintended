from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import pandas as pd

from evaluation.evaluator import Evaluator
from utils.plotting import plot_results, split_violin_plot
from utils.io import load_json
from utils.log import setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', type=str, nargs='+', help="List of folder paths")
    parser.add_argument('--metric', type=str, required=True, help="Metric name to use")

    # Parse the arguments
    args = parser.parse_args()

    outdir = Path(args.folders[0]).parent / "create_sheet_output"
    outdir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(outdir, "exp")
    logger.debug(f"Set up logger in dir '{str(outdir)}'.")

    evaluator = Evaluator(logger, outdir)
    random.seed(97)

    output_file_name = "output.json"

    type_to_pretty_text = {
        "ns": "homophone (0)",
        "ra": "random (0)",
        "sp": "pun syn (0)",
        "sa": "alt syn (0)",
        "pos": "pun (1)",
        "neg": "sentence (0)",
        "test_negative": "PunEval (0)",
    }

    metric_records = list()
    all_metrics = list()

    collected_runs_filename = os.path.join(outdir, f"{args.metric}_runs.csv")

    if not os.path.exists(collected_runs_filename):
        # Results have not been processed previously
        for folder in args.folders:
            print(f"Processing run folder: {folder}")
            for data_name in os.listdir(folder):
                if os.path.isdir(os.path.join(folder, data_name)):
                    for model_name in os.listdir(os.path.join(folder, data_name)):
                        for prompt_name in os.listdir(os.path.join(folder, data_name, model_name)):
                            results_file = os.path.join(folder, data_name, model_name, prompt_name, output_file_name)
                            if os.path.exists(results_file):
                                logger.info("Processing results file '%s'" % results_file)
                                results = load_json(Path(results_file))
                                evaluate_rationale: bool = prompt_name in ["p3", "p5"]

                                # Create e dataset with the homo-graphic examples only
                                hom_ids = {e["id"] for e in results if e["example"]["label"] == 1 and not e["example"]["is_het"]}
                                hom_ids |= {f"{type_str}_{e_id}" for type_str in ["ns", "sa", "sp", "ra"] for e_id in hom_ids}
                                hom_dataset = list()
                                for e in results:
                                    e_id = e["id"]
                                    if e_id in hom_ids:
                                        hom_dataset.append(e)
                                assert len(hom_dataset) == len(hom_ids) == 500, "Wrong number of examples: {}".format(len(hom_dataset))
                                # Keep a version of the HOM dataset without the sampled negatives and the generated negatives (only substitution data)
                                hom_dataset_only_substitution = hom_dataset.copy()

                                # Select the negatives from the TEST dataset (for comparison)
                                folders_splits = folder.split(os.path.sep)
                                base_data_test = Path(os.path.join(*folders_splits[:2], "base_data", folders_splits[-1], "test_puns", model_name, prompt_name)) / output_file_name
                                logger.info("Processing results file '%s'" % base_data_test)
                                sampled_data = list()
                                if os.path.exists(base_data_test):
                                    test_data = load_json(base_data_test)
                                    base_data = [e for e in test_data if e["label"] == 0]
                                    sampled_data = random.sample(base_data, 100)
                                    for e in sampled_data:
                                        # We need to take care of some cases where not all fields are set
                                        e.setdefault("example", dict())
                                        e["example"]["type"] = "test_negative"
                                        e["example"]["id"] = e["id"]
                                        e["example"]["label"] = e["label"]

                                # Add 100 random sentences (negatives) that are in common for HET/HOM examples
                                hom_dataset += [e for e in results if e["example"]["type"] == "neg"]
                                assert len(hom_dataset) == 600, "Wrong number of HOM examples: {}".format(len(hom_dataset))

                                # Add the test negatives
                                hom_dataset.extend(sampled_data)
                                assert len(hom_dataset) == 700, "Wrong number of HOM examples: {}".format(len(hom_dataset))

                                for typ in type_to_pretty_text.keys():
                                    pos_class = 1 if typ == "pos" else 0
                                    positive_dataset = [e["example"] for e in hom_dataset if e["example"]["type"] == typ]
                                    positive_results = [e for e in hom_dataset if e["example"]["type"] == typ]
                                    metrics = evaluator.evaluate_metrics(positive_dataset, positive_results, False, evaluate_rationale, pos_class=pos_class,
                                                                         hom_het_evaluation=False)
                                    pos_metric = metrics[args.metric]
                                    metric_records.append((type_to_pretty_text[typ], pos_metric, folder, prompt_name, model_name, "HOM"))

                                # Hetero-graphic examples are all that are not homographic
                                het_dataset = [e for e in results if e["id"] not in hom_ids]
                                assert len(het_dataset) == 600, "Wrong number of HET examples: {}".format(len(het_dataset))

                                # Keep a version of the HET dataset without the sampled negatives and the generated negatives (only substitution data)
                                het_dataset_only_substitution = het_dataset.copy()
                                het_dataset_only_substitution = [e for e in het_dataset_only_substitution if e["example"]["type"] != "neg"]
                                assert len(het_dataset_only_substitution) == 500, "Wrong number of examples in the reduced HET dataset: {}".format(
                                    len(het_dataset_only_substitution))

                                # Add the sampled data
                                het_dataset.extend(sampled_data)
                                assert len(het_dataset) == 700, "Wrong number of HET examples: {}".format(len(het_dataset))

                                for typ in type_to_pretty_text.keys():
                                    pos_class = 1 if typ == "pos" else 0
                                    positive_dataset = [e["example"] for e in het_dataset if e["example"]["type"] == typ]
                                    positive_results = [e for e in het_dataset if e["example"]["type"] == typ]
                                    metrics = evaluator.evaluate_metrics(positive_dataset, positive_results, False, evaluate_rationale, pos_class=pos_class,
                                                                         hom_het_evaluation=False)
                                    pos_metric = metrics[args.metric]
                                    metric_records.append((type_to_pretty_text[typ], pos_metric, folder, prompt_name, model_name, "HET"))

                                # Overall metrics separated by HET/HOM
                                # HET
                                het_examples = [e["example"] for e in het_dataset_only_substitution]
                                overall_metrics = evaluator.evaluate_metrics(het_examples, het_dataset_only_substitution, True, evaluate_rationale, pos_class=1,
                                                                             hom_het_evaluation=False)
                                df = pd.Series(overall_metrics).to_frame("value")
                                df["run_folder"] = folder
                                df["prompt"] = prompt_name
                                df["model"] = model_name
                                df["type"] = "het"
                                all_metrics.append(df)

                                # HOM
                                hom_examples = [e["example"] for e in hom_dataset_only_substitution]
                                overall_metrics = evaluator.evaluate_metrics(hom_examples, hom_dataset_only_substitution, True, evaluate_rationale, pos_class=1,
                                                                             hom_het_evaluation=False)
                                df = pd.Series(overall_metrics).to_frame("value")
                                df["run_folder"] = folder
                                df["prompt"] = prompt_name
                                df["model"] = model_name
                                df["type"] = "hom"
                                all_metrics.append(df)

        df = pd.DataFrame.from_records(metric_records, columns=["type", args.metric, "folder", "prompt", "model", "data_name"])
        df.to_csv(collected_runs_filename)

        # Overall metrics on each dataset
        runs_average = pd.concat(all_metrics).reset_index().groupby(["model", "type", "index", "prompt"])["value"].agg(mean='mean', std='std').reset_index()
        runs_average.to_csv(os.path.join(outdir, f"{args.metric}_averages.csv"))
    else:
        # Reload the previous file (useful to make pretty plots without waiting 10 minutes each time)
        df = pd.read_csv(collected_runs_filename, usecols=["type", args.metric, "folder", "prompt", "model", "data_name"]).reset_index(drop=True)

    model_names = {"gemini-2.0-flash-001": 'Gemini2.0',
                   "gpt-4o": 'GPT-4o',
                   "llama-3.3-70b-instruct": 'Llama3.3',
                   "mistral-small-24b-instruct-2501": 'Mistral3',
                   "qwen-2.5-72b-instruct": 'Qwen2.5',
                   "deepseek-r1-distill-llama-70b": 'R1-D',
                   "deepseek-reasoner": 'R1'
                   }
    df["model"] = df["model"].replace(model_names)
    df["prompt"] = df["prompt"].replace({"p1": "0s", "p2": "fs", "p3": "w", "p5": "w+s"})

    df["data_name"] = df["data_name"].replace(type_to_pretty_text)

    # PROMPTING PLOTS

    bar_order = type_to_pretty_text.values()

    # **** MODEL RECALL/PRECISION TRUE WITH THE BEST PROMPT OVER PUNBREAK BARPLOT
    df = df.rename(columns={"type": "substitution"})
    model_selection = {'Gemini2.0': 'w+s',
                       'GPT-4o': 'w',
                       'Llama3.3': 'w',
                       'Mistral3': '0s',
                       'Qwen2.5': 'w',
                       'R1-D': 'w+s',
                       'R1': 'w+s'
                       }
    df_filtered = df.copy(deep=True)
    for m, p in model_selection.items():
        df_filtered.loc[df_filtered["model"] == m] = df_filtered.loc[(df_filtered["model"] == m) & (df_filtered["prompt"] == p), :]
    df_filtered["prompt"] = "best"
    plot_results(outdir, df_filtered, x_axis="model", y_axis=args.metric, group_by="substitution",
                 plot_column="prompt",
                 title=f"{args.metric.capitalize()} by substitution best prompt", y_axis_label=args.metric,
                 figsize=(14, 8), plot_type="barplot",
                 hue_order=bar_order, order=model_selection.keys(), custom_legend_on_top=True)  # font_scale=2.7 for POSTER only

    # **** RATIONALE IMPACT ON RECALL/PRECISION TRUE (ALL CATEGORIES)
    df_filtered = df.copy(deep=True)
    df_filtered = df_filtered.loc[(df_filtered["substitution"].isin(["homophone (0)", "random (0)", "pun syn (0)", "alt syn (0)"]))]
    df_filtered["data_name"] = "ALL"

    # Instead of showing all prompts, shows just split violin plot of fs vs best rationale
    rationale_selection = {
        'Gemini2.0': 'w+s',
        'GPT-4o': 'w',
        'Llama3.3': 'w',
        'Mistral3': 'w',
        'Qwen2.5': 'w',
        'R1-D': 'w+s',
        'R1': 'w+s'
    }
    # FILTER BEST PROMPT
    for m, p in rationale_selection.items():
        df_filtered = df_filtered.drop(df_filtered[(df_filtered["model"] == m) & (~df_filtered["prompt"].isin([p, "fs"]))].index)
    df_filtered.loc[:, "prompt"] = df_filtered["prompt"].replace({"w+s": "rationale (w or w+s)", "w": "rationale (w or w+s)"})
    # VIOLIN
    split_violin_plot(outdir, df_filtered, x_axis="model", y_axis=args.metric, group_by="prompt",
                      plot_column="data_name", order=rationale_selection.keys(),
                      plot_type="violin", y_axis_label=f"{args.metric.capitalize()}", title=f"Rationale effect {args.metric} by prompt",
                      figsize=(12, 8), hue_order=["fs", "rationale (w or w+s)"])  # gap=0.2
    # BOX PLOT
    # grouped_box_plot(outdir, df_filtered, x_axis="model", y_axis=args.metric, group_by="prompt",
    #                  plot_column="data_name",
    #                  title=f"Rationale effect by prompt", y_axis_label="{args.metric.capitalize()} (class 0)",
    #                  figsize=(14, 10), plot_type="boxplot",
    #                  hue_order=["0s", "fs", "w", "w+s"], order=model_selection.keys())  # order=['Gemini2.0', 'GPT-4o', 'Llama3.3', 'Qwen2.5', 'DS-R1-d', 'DS-R1']

    # STATISTICS
    df.groupby(["substitution", "prompt", "model", "data_name"])[args.metric].agg(mean="mean", std="std").reset_index().to_csv(
        os.path.join(outdir, f"{args.metric}_statistics.csv"))
    df.groupby(["substitution", "prompt", "model"])[args.metric].agg(mean="mean", std="std").reset_index().to_csv(os.path.join(outdir, f"{args.metric}_statistics_all.csv"))


if __name__ == "__main__":
    main()
