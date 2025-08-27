#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Required now for combined plot styling
import torch
from datasets import Dataset
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  # For progress bars
from transformers import AutoModel, AutoTokenizer, set_seed
from umap import UMAP  # Use 'umap-learn' package

from utils.io import load_json, save_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Set seed for reproducibility (especially for t-SNE/UMAP)
set_seed(42)


# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from a HuggingFace model and visualize class separation using t-SNE and UMAP, both individually per dataset and combined."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the fine-tuned model and tokenizer.",
    )
    parser.add_argument(
        "--test_files",
        nargs="+",  # Allows one or more test files
        type=str,
        required=True,
        help="Path(s) to the test JSON file(s). Each file must contain 'text' and 'label' keys.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output plots.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="The maximum total input sequence length after tokenization. Should match training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization."
    )

    parser.add_argument("--suffix",
                        type=str,
                        default="",
                        help="Suffix for output files")

    args = parser.parse_args()

    # if not os.path.isdir(args.model_dir):
    #     parser.error(f"Model directory not found: {args.model_dir}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    return args


# --- Tokenization Function ---
def tokenize_function(examples: Dict[str, List], tokenizer: AutoTokenizer, max_len: int) -> Dict:
    """Tokenizes text for embedding extraction."""
    # Ensure 'text' field exists and handle potential None values if necessary
    texts = [str(t) if t is not None else "" for t in examples["text"]]
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"  # Return PyTorch tensors
    )


# --- Embedding Extraction ---
def extract_embeddings(
        dataset: Dataset,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        batch_size: int,
        max_seq_length: int,
        device: torch.device,
        mean_pooling: bool = False
) -> np.ndarray:
    """Extracts embeddings (mean pooling of last hidden state) for a dataset."""
    model.eval()  # Set model to evaluation mode
    model.to(device)
    all_embeddings = []

    # Keep only columns needed for tokenization
    dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_seq_length),
        batched=True,
        remove_columns=[col for col in dataset.column_names if col not in ['text']],  # Remove original text after use if desired
        desc="Tokenizing"
    )
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size)

    logging.info(f"Extracting embeddings using device: {device}")
    with torch.no_grad():  # Disable gradient calculations
        for batch in tqdm(dataloader, desc="Extracting Embeddings"):
            # Move batch to device
            inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}

            # Get model outputs
            outputs = model(**inputs)

            # Extract last hidden states
            last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)

            if mean_pooling:
                # --- Mean Pooling ---
                # Get attention mask for averaging (ignore padding tokens)
                attention_mask = inputs['attention_mask']  # Shape: (batch_size, seq_length)
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                # Sum embeddings * mask
                sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
                # Sum mask elements to get actual sequence lengths (avoid division by zero)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                # Calculate mean pooled embeddings
                embeddings = sum_embeddings / sum_mask
                # --- End Mean Pooling ---
            else:
                # Alternative: CLS Token Embedding
                embeddings = last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)

            all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


# --- Dimensionality Reduction (Helper) ---
def run_dimensionality_reduction(
        embeddings: np.ndarray,
        method: str,  # 'tsne' or 'umap'
        seed: int
) -> np.ndarray:
    """Performs t-SNE or UMAP reduction."""
    logging.info(f"Performing {method.upper()} dimensionality reduction on {embeddings.shape[0]} samples (may take time)...")
    if method == 'tsne':
        reducer = TSNE(
            n_components=2,
            random_state=seed,
            perplexity=min(30, len(embeddings) - 1),  # Perplexity must be less than n_samples
            n_iter=1000,
            init='pca',  # Faster and often more stable initialization
            learning_rate='auto'
        )
    elif method == 'umap':
        reducer = UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=min(15, len(embeddings) - 1),  # n_neighbors must be less than n_samples
            min_dist=0.1,
            metric='cosine'  # Cosine distance often works well for text embeddings
        )
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    embeddings_2d = reducer.fit_transform(embeddings)
    logging.info(f"{method.upper()} reduction complete.")
    return embeddings_2d


# --- Plotting Function (Individual Datasets) ---
# Renamed for clarity
def plot_individual(
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        method: str,
        output_path: str,
        dataset_name: str
):
    """Saves a scatter plot for an individual dataset."""
    plt.figure(figsize=(10, 7))
    title = f'{method.upper()} Visualization for {dataset_name}'
    marker_map = {0: '^', 1: 'o'}
    color_palette = {0: "black", 1: "orange"}

    # Use seaborn for potentially nicer plots and automatic legend
    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=labels,
        markers=marker_map,
        palette=color_palette,  # Use distinct colors
        alpha=0.7,
        style=labels,
        legend='full',
        s=50  # marker size
    )  # .set_title(title)

    plt.xticks([])  # Remove x ticks
    plt.yticks([])  # Remove y ticks
    plt.legend(fontsize='xx-large', title_fontsize='40').set_visible(False)

    # plt.xlabel(f'{method.upper()} Dimension 1')
    # plt.ylabel(f'{method.upper()} Dimension 2')
    plt.grid(False)
    logging.info(f"Saving individual {method.upper()} plot to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory


# --- Plotting Function (Combined Datasets) ---
# NEW function
def plot_combined(
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        dataset_ids: np.ndarray,  # Array indicating source dataset index (0, 1, ...)
        dataset_names: List[str],  # List of actual dataset names for legend
        method: str,
        output_path: str
):
    """Saves a scatter plot combining all data with distinct markers."""
    num_datasets = len(dataset_names)
    # Define markers - add more if you have more data
    markers = ['o', 's', '^', 'X', 'D', 'v', '<', '>', 'P', '*']
    if num_datasets > len(markers):
        logging.warning(f"More data ({num_datasets}) than defined markers ({len(markers)}). Markers will repeat.")
        # Extend markers by repeating if necessary
        markers = (markers * (num_datasets // len(markers) + 1))[:num_datasets]

    plt.figure(figsize=(12, 10))  # Slightly larger figure for combined plot
    title = f'Combined {method.upper()} Visualization (Color=Class, Marker=Dataset)'

    # Use seaborn's 'style' parameter for different markers based on dataset_ids
    # Map numerical dataset_ids to actual names for the legend
    dataset_id_map = {i: name for i, name in enumerate(dataset_names)}
    dataset_col = np.vectorize(dataset_id_map.get)(dataset_ids)  # Convert IDs to names for Seaborn legend

    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=labels,  # Color by class label (0/1)
        style=dataset_col,  # Use different markers for different data (using mapped names)
        markers=markers[:num_datasets],  # Provide the list of markers to use
        palette=sns.color_palette("bright", n_colors=len(np.unique(labels))),
        alpha=0.7,
        s=60  # Slightly larger markers for combined plot
    ).set_title(title)

    # plt.xlabel(f'{method.upper()} Dimension 1')
    # plt.ylabel(f'{method.upper()} Dimension 2')
    # Enhance legend positioning if needed (optional)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.5)
    logging.info(f"Saving combined {method.upper()} plot to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Use bbox_inches='tight' to prevent legend cutoff
    plt.close()


def cluster_and_visualize_dbscan(
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        texts: List[str],
        method: str,  # e.g., 'tsne' or 'umap' - used for filenames/titles
        output_base_path: str,  # Base path for output files (e.g., /path/to/output/dataset_name)
        dataset_name: str,
        dbscan_eps: float = 0.5,  # DBSCAN eps parameter - may require tuning!
        dbscan_min_samples: int = 5  # DBSCAN min_samples parameter - may require tuning!
):
    """
    Performs DBSCAN clustering on 2D embeddings, saves text examples per cluster
    to JSON, and creates a scatter plot visualizing clusters (color) and
    original labels (marker).

    Args:
        embeddings_2d: NumPy array of shape (n_samples, 2) with 2D embeddings.
        labels: NumPy array of shape (n_samples,) with original binary labels (0/1).
        texts: List of strings (n_samples,) with original text data.
        method: String indicating the dimensionality reduction method used ('tsne'/'umap').
        output_base_path: Base path string for saving output files (plot and JSON).
        dataset_name: Name of the dataset for titles.
        dbscan_eps: The epsilon parameter for DBSCAN.
        dbscan_min_samples: The min_samples parameter for DBSCAN.
    """
    logging.info(f"Starting DBSCAN clustering for {dataset_name} ({method})...")

    # --- Input Validation ---
    n_samples = embeddings_2d.shape[0]
    if not (len(labels) == n_samples and len(texts) == n_samples):
        logging.error(f"Mismatch in input lengths: embeddings ({n_samples}), "
                      f"labels ({len(labels)}), texts ({len(texts)})")
        return

    # --- Data Scaling (recommended for DBSCAN) ---
    # Scale features to have zero mean and unit variance
    # This makes the choice of `eps` less dependent on the absolute scale of embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_2d)
    logging.info(f"Embeddings scaled using StandardScaler.")

    # --- DBSCAN Clustering ---
    try:
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="euclidean", n_jobs=-1)  # Use all CPU cores
        cluster_labels = dbscan.fit_predict(embeddings_scaled)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        logging.info(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points.")
        if n_clusters == 0:
            logging.warning("DBSCAN did not find any clusters. All points may be noise. "
                            "Consider adjusting `eps` and `min_samples`.")

    except Exception as e:
        logging.error(f"DBSCAN clustering failed: {e}")
        return

    # --- Prepare JSON Output ---
    clusters_dict = dict()
    for i, cluster_id in enumerate(cluster_labels):
        # Ensure keys are strings for JSON compatibility, although ints often work
        cluster_key = str(cluster_id)
        if cluster_key not in clusters_dict:
            clusters_dict[cluster_key] = []
        clusters_dict[cluster_key].append(texts[i])

    # --- Save JSON ---
    try:
        save_json(Path(f"{output_base_path}_dbscan_{method}_clusters.json"), clusters_dict)
    except IOError as e:
        logging.error(f"Failed to write JSON file: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while saving JSON: {e}")

    # --- Create Plot ---
    plot_output_path = f"{output_base_path}_dbscan_{method}_plot.png"
    logging.info(f"Creating DBSCAN visualization plot: {plot_output_path}")

    plt.figure(figsize=(12, 10))
    title = f'DBSCAN Clustering ({method.upper()}) for {dataset_name}\n(Color=Cluster, Marker=Original Label)'

    # Define markers for original labels
    markers = {0: "o", 1: "X"}  # e.g., Circle for label 0, X for label 1

    # Define a palette. Use a qualitative one if few clusters, sequential/diverging if many.
    # Include gray for noise points (cluster label -1)
    unique_cluster_labels = sorted(list(set(cluster_labels)))
    n_colors = len(unique_cluster_labels)
    # Use 'tab10' or 'tab20' for fewer clusters, 'viridis'/'plasma' etc. for many
    # Handle noise (-1) separately with a specific color (e.g., gray)
    palette_name = "viridis" if n_colors > 10 else "tab10"
    # Generate colors, reserving one for noise if present
    num_actual_clusters = n_colors - 1 if -1 in unique_cluster_labels else n_colors
    colors_for_clusters = sns.color_palette(palette_name, num_actual_clusters)

    # Create color mapping: cluster_id -> color
    color_map = {}
    color_idx = 0
    for cl_id in unique_cluster_labels:
        if cl_id == -1:
            color_map[cl_id] = (0.5, 0.5, 0.5, 0.6)  # Gray for noise (with some transparency)
        else:
            if color_idx < len(colors_for_clusters):
                color_map[cl_id] = colors_for_clusters[color_idx]
                color_idx += 1
            else:  # Fallback if somehow palette is too small
                color_map[cl_id] = (0.0, 0.0, 0.0, 0.6)  # Black fallback

    # Map cluster and original labels to plot arguments
    # point_colors = [color_map[cl] for cl in cluster_labels]
    # point_markers = [markers[lab] for lab in labels]  # Need to plot marker by marker type

    # Plotting with seaborn's scatterplot is cleaner for combined legends
    # Prepare data for seaborn
    plot_data = {
        f'{method.upper()} Dim 1': embeddings_2d[:, 0],
        f'{method.upper()} Dim 2': embeddings_2d[:, 1],
        'Cluster': [f'Cluster {c}' if c != -1 else 'Noise' for c in cluster_labels],  # Use strings for legend
        'Original Label': labels  # Keep original labels 0/1
    }

    try:
        ax = sns.scatterplot(
            data=plot_data,
            x=f'{method.upper()} Dim 1',
            y=f'{method.upper()} Dim 2',
            hue='Cluster',  # Color by DBSCAN cluster label
            style='Original Label',  # Use different markers for original labels
            hue_order=[f'Cluster {c}' for c in sorted([lbl for lbl in unique_cluster_labels if lbl != -1])] + (['Noise'] if -1 in unique_cluster_labels else []),
            # Control legend order
            palette=[color_map[cl] for cl in sorted([lbl for lbl in unique_cluster_labels if lbl != -1])] + ([color_map[-1]] if -1 in unique_cluster_labels else []),
            # Provide exact colors in order
            markers=markers,  # Specify markers for style levels (0 and 1)
            s=50,  # Marker size
            alpha=0.7
        )
        ax.set_title(title)
        plt.grid(True, linestyle='--', alpha=0.5)
        # Adjust legend position
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        logging.info(f"Saving DBSCAN plot to {plot_output_path}")
        plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')

    except Exception as e:
        logging.error(f"Failed to create or save plot: {e}")
    finally:
        plt.close()  # Close the figure


def optimal_eps(embeddings, method: str):
    """
    Utility to find best EPS parameter for DBSCAN. k should be set to min_samples (min points)
    Taken from: https://stataiml.com/posts/how_to_set_dbscan_paramter
    """
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    import matplotlib.pyplot as plt

    # initialize the value of k for kNN which can be same as MinPts
    k = 8

    # Compute k-nearest neighbors
    # you need to add 1 to k as this function also return
    # distance to itself (first column is zero)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)

    # get distances
    dist, ind = nbrs.kneighbors(embeddings)

    k_dist = np.sort(dist[:, -1])

    plt.plot(k_dist)
    plt.xlabel('Distance sorted points')
    plt.ylabel(f'{k}-Distance')
    plt.title(method)
    plt.grid()
    plt.show()


# --- Main Function ---
def main():
    args = parse_arguments()

    # --- Determine Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Model and Tokenizer ---
    logging.info(f"Loading model and tokenizer from {args.model_dir}")
    try:
        model = AutoModel.from_pretrained(args.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    except Exception as e:
        logging.error(f"Error loading model or tokenizer from {args.model_dir}: {e}")
        sys.exit(1)

    # --- Data storage for combined plots ---
    all_embeddings_list = []
    all_labels_list = []
    all_dataset_indices_list = []
    dataset_names = []

    # --- Process Each Test File ---
    for i, test_file_path in enumerate(args.test_files):
        dataset_name = Path(test_file_path).stem
        dataset_names.append(dataset_name)  # Store name for combined legend
        logging.info(f"--- Processing dataset {i + 1}/{len(args.test_files)}: {dataset_name} ({test_file_path}) ---")

        # --- Load Data ---
        test_data = load_json(Path(test_file_path))
        if not test_data:
            logging.warning(f"No data loaded from {test_file_path}, skipping.")
            continue

        # Extract labels and texts
        try:
            labels = np.array([item['label'] for item in test_data])
            texts = [item['text'] for item in test_data]
        except KeyError as e:
            logging.error(f"Missing key {e} in file {test_file_path}. Ensure 'text' and 'label' exist.")
            continue

        # Create Hugging Face Dataset
        raw_test_dataset = Dataset.from_dict({"text": texts})

        # --- Extract Embeddings ---
        embeddings = extract_embeddings(
            raw_test_dataset, model, tokenizer, args.batch_size, args.max_seq_length, device, mean_pooling=True
        )
        logging.info(f"Extracted {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}.")

        # --- Store data for combined plot ---
        all_embeddings_list.append(embeddings)
        all_labels_list.append(labels)
        # Assign a unique index 'i' to all samples from this dataset
        all_dataset_indices_list.append(np.full(embeddings.shape[0], i, dtype=int))

        # --- Perform Individual Dimensionality Reduction and Plotting ---
        # t-SNE Individual
        embeddings_2d_tsne_indiv = run_dimensionality_reduction(embeddings, 'tsne', args.seed)
        tsne_output_path_indiv = os.path.join(args.output_dir, f"{dataset_name}_tsne_separation{args.suffix.strip()}.png")
        plot_individual(embeddings_2d_tsne_indiv, labels, 'tsne', tsne_output_path_indiv, dataset_name)

        # UMAP Individual
        # embeddings_2d_umap_indiv = run_dimensionality_reduction(embeddings, 'umap', args.seed)
        # umap_output_path_indiv = os.path.join(args.output_dir, f"{dataset_name}_umap_separation.png")
        # plot_individual(embeddings_2d_umap_indiv, labels, 'umap', umap_output_path_indiv, dataset_name)

        logging.info(f"--- Finished individual processing for dataset: {dataset_name} ---")

        if dataset_name == "test_puns":
            # This part only to plot and find "elbow" to set eps
            # optimal_eps(embeddings_2d_umap_indiv, "umap")
            # optimal_eps(embeddings_2d_tsne_indiv, "tsne")
            # exit(3)

            # base_path = os.path.join(args.output_dir, dataset_name)
            # logging.info(f"--- Running DBSCAN analysis for {dataset_name} ---")
            # # Run DBSCAN on t-SNE results
            # cluster_and_visualize_dbscan(
            #     embeddings_2d=embeddings_2d_tsne_indiv,  # Use the 2D embeddings from t-SNE
            #     labels=labels,
            #     texts=texts,  # Make sure you have the original texts list available here
            #     method='tsne',
            #     output_base_path=base_path,
            #     dataset_name=dataset_name,
            #     dbscan_eps=0.25,  # Adjust these parameters as needed!
            #     dbscan_min_samples=4
            # )

            # Run DBSCAN on UMAP results
            # cluster_and_visualize_dbscan(
            #     embeddings_2d=embeddings_2d_umap_indiv,  # Use the 2D embeddings from UMAP
            #     labels=labels,
            #     texts=texts,  # Make sure you have the original texts list available here
            #     method='umap',
            #     output_base_path=base_path,
            #     dataset_name=dataset_name,
            #     dbscan_eps=0.25,  # Adjust these parameters as needed!
            #     dbscan_min_samples=4
            # )
            logging.info(f"--- Finished DBSCAN analysis for {dataset_name} ---")
            # exit(2)

    # --- Combined Processing (if more than one dataset) ---
    # if len(args.test_files) > 0 and len(all_embeddings_list) > 0:
    #     logging.info("--- Starting combined dataset processing ---")
    #     # Concatenate all collected data
    #     combined_embeddings = np.concatenate(all_embeddings_list, axis=0)
    #     combined_labels = np.concatenate(all_labels_list, axis=0)
    #     combined_dataset_indices = np.concatenate(all_dataset_indices_list, axis=0)
    #     logging.info(f"Total samples for combined plot: {combined_embeddings.shape[0]}")
    #
    #     # --- Combined t-SNE Reduction and Plotting ---
    #     combined_embeddings_2d_tsne = run_dimensionality_reduction(combined_embeddings, 'tsne', args.seed)
    #     tsne_output_path_combined = os.path.join(args.output_dir, "combined_tsne_separation.png")
    #     plot_combined(
    #         combined_embeddings_2d_tsne,
    #         combined_labels,
    #         combined_dataset_indices,
    #         dataset_names,  # Pass names for legend
    #         'tsne',
    #         tsne_output_path_combined
    #     )
    #
    #     # --- Combined UMAP Reduction and Plotting ---
    #     combined_embeddings_2d_umap = run_dimensionality_reduction(combined_embeddings, 'umap', args.seed)
    #     umap_output_path_combined = os.path.join(args.output_dir, "combined_umap_separation.png")
    #     plot_combined(
    #         combined_embeddings_2d_umap,
    #         combined_labels,
    #         combined_dataset_indices,
    #         dataset_names,  # Pass names for legend
    #         'umap',
    #         umap_output_path_combined
    #     )
    #     logging.info("--- Finished combined dataset processing ---")
    #
    # elif len(args.test_files) > 0:
    #     logging.warning("No valid data found in any input file to create combined plots.")
    # else:
    #     logging.info("No input files provided.")

    logging.info("All processing finished. Script finished.")


if __name__ == "__main__":
    main()
