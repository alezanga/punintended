from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from utils.plotting import plot_results_sp, grouped_box_plot, split_violin_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', type=str, nargs='+', help="List of folder paths")
    parser.add_argument('--prompting', action="store_true",
                        help="results for prompting. If not set, assumes fine tuning")

    # Parse the arguments
    args = parser.parse_args()

    dfs = list()
    for folder in args.folders:
        print(f"Processing run folder: {folder}")
        for data_name in os.listdir(folder):
            if os.path.isdir(os.path.join(folder, data_name)):
                for model_name in os.listdir(os.path.join(folder, data_name)):
                    for prompt_name in os.listdir(os.path.join(folder, data_name, model_name)):
                        results_file = os.path.join(folder, data_name, model_name, prompt_name, "metrics.json")

                        if os.path.exists(results_file):
                            ds = pd.read_json(results_file, typ="series")
                            ds = pd.DataFrame(ds, columns=["value"])

                            ds["run_folder"] = folder
                            ds["prompt"] = prompt_name
                            ds["data"] = data_name
                            ds["model"] = model_name
                            dfs.append(ds)

    outdir = Path(args.folders[0]).parent / "create_sheet_output"
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.concat(dfs)
    df.to_csv(os.path.join(outdir, "runs.csv"))
    df = df[df['data'] != 'val_puns']
    # De-normalize the agreement metric
    df.loc["kw_agreement", "value"] *= 2

    for value in df.index.unique():
        df_m = df[df.index == value].copy()
        runs_average = df_m.groupby(["prompt", "data", "model"])["value"].agg(mean='mean', std='std').reset_index()

        filtered_runs_average = runs_average[~runs_average['data'].isin(['new', 'test_puns'])]
        runs_average_all = filtered_runs_average.groupby(["prompt", "model"])["mean"].agg(mean='mean', std='std').reset_index()
        runs_average_all["data"] = "ALL"  # Add a placeholder for the 'data' column

        final_output = pd.concat([runs_average, runs_average_all], ignore_index=True).round(4)
        if value == "kw_agreement":
            final_output["mean"] = final_output["mean"].round(1)
        final_output.sort_values(by=["data", "model"]).to_csv(os.path.join(outdir, f"avg_{str(value)}.csv"),
                                                              index=False)
    # order = ["p5.txt", "p3.txt", "p1.txt"]

    prompts_pretty_names = {"p1": "0s", "p2": "fs", "p3": "w", "p5": "w+s"}
    df["prompt"] = df["prompt"].replace(prompts_pretty_names)
    df["data"] = df["data"].replace({"test_puns": "PunEval", "new": "base"})
    df["model"] = df["model"].replace({
        "llama-3.3-70b-instruct": "Llama3.3",
        "deepseek-r1-distill-llama-70b": "R1-D",
        "deepseek-reasoner": "R1",
        "gemini-2.0-flash-001": "Gemini2.0",
        "mistral-small-24b-instruct-2501": "Mistral3",
        "qwen-2.5-72b-instruct": "Qwen2.5",
        "gpt-4o": "GPT-4o"
    })

    # df = df.reset_index(drop=False)
    # unbiased_dataset = df.loc[df["data"].isin(["daughter", "doctor", "never_die", "used"])].copy()
    # # averaged_unbiased = unbiased_dataset.groupby(["prompt", "model"])["value"].agg(mean='mean', std='std').reset_index()
    # unbiased_dataset["data"] = "UNBIASED"
    # df = pd.concat([df, unbiased_dataset], ignore_index=True)

    ### PROMPTING PLOTS

    df = df[df["model"] != "llama-3.1-405b-instruct"]
    # df = df[df["model"] != "deepseek-r1-distill-llama-70b"]

    model_selection = {'Gemini2.0': 'w+s',
                       'GPT-4o': 'w',
                       'Llama3.3': 'w',
                       'Mistral3': '0s',
                       'Qwen2.5': 'w',
                       'R1-D': 'w+s',
                       'R1': 'w+s'
                       }

    # NAP + JOKER TREND (boxplot_f1_distribution_nap_+_joker_dataset)
    df_filtered = df.loc[df["data"].isin(['base', 'ruined']), :].reset_index(drop=False)
    grouped_box_plot(outdir, df_filtered, x_axis="model", y_axis="value", group_by="prompt",
                     plot_column="index", plot_values=["f1"],
                     plot_type="boxplot", y_axis_label="index", title="Distribution NAP + JOKER dataset",
                     figsize=(14, 6), order=model_selection.keys(), hue_order=prompts_pretty_names.values())

    # UNBIASED DATA TREND (boxplot_f1_distribution_unbiased_dataset)
    df_filtered = df.loc[df["data"].isin(["daughter", "doctor", "never_die", "used"]), :].reset_index(drop=False)
    grouped_box_plot(outdir, df_filtered, x_axis="model", y_axis="value", group_by="prompt",
                     plot_column="index", plot_values=["f1"],
                     plot_type="boxplot", y_axis_label="index", title="Distribution UNBIASED dataset",
                     figsize=(14, 6), order=model_selection.keys(), hue_order=prompts_pretty_names.values())

    # HET/HOM SPLIT (UNUSED)
    # df_filtered = df.copy()
    # for m, p in model_selection.items():
    #     df_filtered = df_filtered.drop(df_filtered[(df_filtered["model"] == m) & (df_filtered["prompt"] != p) & (df_filtered["data"].isin["ruined"])].index)
    #
    # split_violin_plot(outdir, df_filtered, x_axis="model", y_axis="value", group_by="prompt",
    #                   plot_column="index", plot_values=["f1"],
    #                   plot_type="boxplot", y_axis_label="index", title="Distribution UNBIASED dataset",
    #                   figsize=(14, 6), order=model_selection.keys(), hue_order=prompts_pretty_names.values())

    # UNBIASED RATIONALE TREND PER PUNNY-PATTERN SUBSET (violin_precision_prompt_distribution_biased_subsets)
    rationale_selection = {
        'Gemini2.0': 'w+s',
        'GPT-4o': 'w',
        'Llama3.3': 'w',
        'Mistral3': 'w',
        'Qwen2.5': 'w',
        'R1-D': 'w+s',
        'R1': 'w+s'
    }
    df_filtered = df.loc[df["data"].isin(["daughter", "doctor", "never_die", "used"]), :].reset_index(drop=False)
    for m, p in rationale_selection.items():
        df_filtered = df_filtered.drop(df_filtered[(df_filtered["model"] == m) & (~df_filtered["prompt"].isin([p, "fs"]))].index)
    df_filtered.loc[:, "prompt"] = df_filtered["prompt"].replace({"w+s": "rationale (w or w+s)", "w": "rationale (w or w+s)"})
    mean_metric = "recall"
    mean_stat = df_filtered.loc[df_filtered["index"] == mean_metric].groupby(["prompt"])["value"].mean()
    print(f"Increase in {mean_metric}:", round((mean_stat["rationale (w or w+s)"] - mean_stat["fs"]) / mean_stat["fs"] * 100, 1))
    split_violin_plot(outdir, df_filtered, x_axis="model", y_axis="value", group_by="prompt",
                      plot_column="index", plot_values=["precision"], order=rationale_selection.keys(),
                      plot_type="violin", y_axis_label="index", title="Prompt distribution biased subsets",
                      figsize=(12, 8), vertical=True, hue_order=["fs", "rationale (w or w+s)"], showticklabels=False, showlegend=True)  # gap=0.2

    split_violin_plot(outdir, df_filtered, x_axis="model", y_axis="value", group_by="prompt",
                      plot_column="index", plot_values=["recall"], order=rationale_selection.keys(),
                      plot_type="violin", y_axis_label="index", title="Prompt distribution biased subsets",
                      figsize=(12, 8), vertical=True, hue_order=["fs", "rationale (w or w+s)"], showticklabels=True, showlegend=False)

    # BARPLOT UNBIASED VS PUNEVAL (barplot_unbiased_vs_puneval_dataset)
    hue_order = ["daughter", "doctor", "never_die", "tom", "used", "when", "PunEval"]

    # for prompt in ["words+senses", "words", "base+fs", "base"]:
    df_filtered = df.loc[df["data"].isin(hue_order), :].reset_index(drop=False)
    assert set(df_filtered["data"].unique().tolist()) == set(hue_order)
    df_filtered.rename(columns={"data": "pattern or data"}, inplace=True)

    for m, p in model_selection.items():
        df_filtered.loc[df_filtered["model"] == m] = df_filtered.loc[(df_filtered["model"] == m) & (df_filtered["prompt"] == p), :]
    # df_filtered = df_filtered.reset_index(drop=False)
    plot_results_sp(outdir, df_filtered, x_axis="model", y_axis="value", group_by="pattern or data",
                    plot_column="index", plot_values=["precision", "recall"],  # recall_0
                    title=f"Unbiased vs PunEval dataset", y_axis_label="index",
                    figsize=(10, 6), plot_type="barplot",
                    hue_order=hue_order, order=model_selection.keys())


if __name__ == "__main__":
    main()
