import argparse
import json
import os
from pathlib import Path

import pandas as pd

from utils.plotting import grouped_box_plot, plot_results
from utils.io import load_json


def plot_confidence_distribution(base_dir, out_dir: Path):
    """
    Generates a boxplot of confidence distributions for different models.

    Searches for output.json files within p5 subdirectories of each model
    folder in the base_dir, extracts confidence values, and plots them
    using seaborn. Saves the plot in the base_dir.

    Args:
        base_dir (str): The base directory containing model result folders.
    """
    all_data = []
    model_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    print(f"Found model folders: {model_folders}")

    model_metrics = dict()
    for model_name in model_folders:
        p5_dir = os.path.join(base_dir, model_name, 'p5')
        output_json_path = os.path.join(p5_dir, 'parsed_output.json')
        metrics_file_path = Path(os.path.join(p5_dir, 'metrics_0.json'))

        if os.path.exists(output_json_path):
            print(f"Processing {output_json_path} for model {model_name}...")
            try:
                with open(output_json_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'confidence' in item:
                                try:
                                    confidence = float(item['confidence'])
                                    pred_label = int(item["pred_label"])
                                    true_label = int(item["example"]["label"])

                                    if pred_label == true_label:
                                        error = "TP" if true_label == 1 else "TN"
                                    elif pred_label == 0:  # true_label is 1
                                        error = "FN"
                                    else:
                                        error = "FP"

                                    all_data.append({'Model': model_name, 'Confidence': confidence, "Category": item["example"]["type"], "Predicted Label": pred_label,
                                                     "True Label": true_label, "Prediction Categories": error})
                                except (ValueError, TypeError):
                                    print(f"Warning: Invalid confidence value found in {output_json_path}: {item.get('confidence')}")
                            else:
                                print(f"Warning: Item in {output_json_path} is not a dict or lacks 'confidence' key: {item}")
                    else:
                        print(f"Warning: Content of {output_json_path} is not a list.")
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {output_json_path}")
            except Exception as e:
                print(f"An error occurred processing {output_json_path}: {e}")
        else:
            print(f"Skipping model {model_name}: {output_json_path} not found.")

        model_metrics[model_name] = load_json(metrics_file_path)

    if not all_data:
        print("No confidence data found to plot.")
        return

    df = pd.DataFrame(all_data)
    df["data_name"] = "c"

    model_pretty_names = {
        # "gemini-2.0-flash-001": 'Gemini2.0',
        "gpt-4o": 'GPT-4o',
        "llama-3.3-70b-instruct": 'Llama3.3',
        # "mistral-small-24b-instruct-2501": 'Mistral3',
        "qwen-2.5-72b-instruct": 'Qwen2.5',
        # "deepseek-r1-distill-llama-70b": 'DS-R1 (D)',
        # "deepseek-reasoner": 'DS-R1'
    }

    type_to_pretty_text = {
        "ns": "phonetic (0)",
        "ra": "random (0)",
        "sp": "pun syn (0)",
        "sa": "alt syn (0)",
        "pos": "pun (1)",
        "neg": "sentence (0)"
    }

    df["Model"] = df["Model"].replace(model_pretty_names)
    df["Category"] = df["Category"].replace(type_to_pretty_text)

    df_filtered = df[df['Category'].isin(['phonetic (0)', "random (0)", "pun syn (0)", "alt syn (0)", "pun (1)", "sentence (0)"])]
    plot_results(out_dir, df_filtered, x_axis="Model", y_axis="Confidence", group_by="Prediction Categories",
                 plot_column="data_name",
                 title=f"Correlation confidence and prediction class", y_axis_label="Confidence",
                 figsize=(9, 6), plot_type="boxplot", order=model_pretty_names.values(),
                 showfliers=False, hue_order=["TP", "TN", "FP", "FN"], legend_position="lower center", x_tick_rotation=0, font_scale=2.0)
    # USE BELOW CODE TO USE PLOTLY VERSION
    # grouped_box_plot(out_dir, df_filtered, x_axis="Model", y_axis="Confidence", group_by="Prediction Categories",
    #                  plot_column="data_name",
    #                  title=f"Correlation confidence and prediction class", y_axis_label="Confidence",
    #                  figsize=(9, 6), plot_type="boxplot", order=model_pretty_names.values(),
    #                  showfliers=False, hue_order=["TP", "TN", "FP", "FN"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=str, help="Path to run with confidence values")
    parser.add_argument('--output_dir', type=str, help="Path where to save plots")
    args = parser.parse_args()
    # Define the base directory where model results are stored
    results_base_dir = args.run
    out_dir = Path(args.output_dir)

    # Ensure the base directory exists
    if not os.path.isdir(results_base_dir):
        print(f"Error: Base directory not found: {results_base_dir}")
    else:
        plot_confidence_distribution(results_base_dir, out_dir)
