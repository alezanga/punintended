from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Any, Dict

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from plotly import graph_objects as go, express as px

CB_PALETTE = ["#332288", "#117733", "#44AA99", "#DDCC77", "#88CCEE", "#CC6677", "#AA4499", "#882255"]
seaborn_palette: list = CB_PALETTE


def plot_results(outdir: Path, df: pd.DataFrame, x_axis: str, y_axis: str, group_by: str, plot_column: str,
                 figsize=(12, 8), plot_values: List[str] = None, plot_type: str = "boxplot",
                 y_axis_label: str = None, title: str = None, prefix: str = "",
                 legend_position: str = "best",
                 x_tick_rotation: int = 45,
                 font_scale=2.5,
                 custom_legend_on_top: bool = False, **kwargs):
    """
    Create boxplot or barplots with data

    :param outdir: The folder where to save the PNG file; will be created if not existing.
    :param df: Pandas DataFrame with data to plot
    :param x_axis: The column in the DataFrame to plot on the x-axis
    :param y_axis: The column to plot on the y-axis
    :param group_by: The column to be used in the legend
    :param plot_column: The column that will be iterated over to create different plots
    :param figsize: tuple with figure size for the plot
    :param plot_values: The values of "plot_column" that should be used. A different plot for each of these values will be created.
    :param plot_type: 'barplot' or 'boxplot'
    :param y_axis_label: str to use on the y_axis. If this label is set equal to 'plot_column', the values from that column will be used
    :param title: title of the plot, that will be used to name the file
    :param prefix: a prefix to pre-pend to the output file
    :param kwargs: key-value arguments passed to seaborn
    :return: None
    """
    # Create a boxplot for each model
    if plot_values is None:
        plot_values = df[plot_column].unique().tolist()
    for value in plot_values:
        df_metric = df[df[plot_column] == value]

        if df_metric.empty:
            print(f"No data available for {value}. Skipping...")
            continue

        # for model_name, group in df.groupby(metric):
        plt.figure(figsize=figsize)
        sns.set_theme(font_scale=font_scale)
        sns.set_style("whitegrid")

        df_metric = df_metric.sort_values(by=[x_axis, group_by])

        # Create the boxplot
        if plot_type == "boxplot":
            ax = sns.boxplot(x=x_axis, y=y_axis, hue=group_by, data=df_metric, palette=seaborn_palette, notch=False, orient="v",
                             showmeans=True,
                             meanprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black", markersize="6"),
                             medianprops=dict(color="black", label="_median_", linewidth=2),
                             flierprops=dict(alpha=0.3),
                             **kwargs)
        elif plot_type == "barplot":
            ax = sns.barplot(x=x_axis, y=y_axis, hue=group_by, data=df_metric, palette=seaborn_palette, errorbar="sd",
                             orient="v", **kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        out_file = f"{prefix}{plot_type}_{value}"
        if title is not None:
            out_file += f"_{title.lower().replace(' ', '_')}"
        out_file += ".png"

        y_axis_str: str = value if (y_axis_label == plot_column or y_axis_label is None) else y_axis_label
        plot_title: str = value if title is None else title
        # plt.title(plot_title)
        plt.xlabel("")
        plt.ylabel(y_axis_str.capitalize())
        plt.xticks(rotation=x_tick_rotation)

        if custom_legend_on_top:
            # MAGIC CODE TO PRODUCE PLOT FOR -- **** MODEL RECALL TRUE WITH THE BEST PROMPT OVER PUNBREAK BARPLOT (phonetic script)
            rect_top = 1.0
            num_legend_rows = 1
            legend_height_factor = 0.01 * num_legend_rows
            rect_top -= (legend_height_factor + 0.01)
            rect_top += 0.08
            # *** PAPER PLOT
            plt.legend(title=group_by.capitalize(), ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.34), handletextpad=0.1, borderaxespad=0.1, labelspacing=0.05)
            plt.tight_layout(pad=0.2, rect=[0.0, 0.0, 1.0, rect_top])
            # *** POSTER PLOT ONLY (MAGNIFY)
            # plt.legend(title=group_by.capitalize(), ncol=4, loc="upper center", bbox_to_anchor=(0.45, 1.34), handletextpad=0.1, borderaxespad=0.01,
            #            labelspacing=0.02, columnspacing=0.2, title_fontsize=30)
            # plt.tight_layout(pad=0.1, rect=[0.0, 0.0, 1.0, rect_top])
            # plt.subplots_adjust(left=0.1, right=0.99)
        else:
            plt.legend(title=group_by.capitalize(), ncol=2, loc=legend_position)
            plt.tight_layout(pad=0.2)

        # Save the figure
        outdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(outdir / out_file, dpi=400)
        plt.clf()
        plt.close()
        # plt.show()


def plot_results_sp(
        outdir: Path,
        df: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        group_by: str,
        plot_column: str,
        figsize: tuple = (12, 8),
        plot_values: Optional[List[str]] = None,
        plot_type: str = "boxplot",
        y_axis_label: Optional[str] = None,
        title: Optional[str] = None,
        prefix: str = "",
        vertical: bool = True,
        **kwargs: Any
) -> None:
    """
    Create combined (stacked) boxplots or barplots from a DataFrame.

    :param outdir: The folder where to save the PNG file; will be created if not existing.
    :param df: Pandas DataFrame with data to plot.
    :param x_axis: The column in the DataFrame to plot on the x-axis.
    :param y_axis: The column to plot on the y-axis.
    :param group_by: The column to be used in the legend.
    :param plot_column: The column that will be iterated over to create different subplots.
    :param figsize: tuple with figure size. This is the base size for one subplot
                    (width is kept, height is scaled by the number of subplots).
    :param plot_values: The values of "plot_column" that should be used for subplots.
                        Defaults to all unique, sorted, non-NaN values.
    :param plot_type: 'barplot' or 'boxplot'.
    :param y_axis_label: str to use on the y_axis of subplots.
                         If set to 'plot_column', values from plot_column are used.
                         If None, defaults to using plot_column values.
    :param title: Overall title for the entire figure (suptitle).
    :param prefix: A prefix to pre-pend to the output file.
    :param kwargs: Additional key-value arguments passed to the seaborn plotting function.
    :return: None
    """
    if plot_values is None:
        plot_values = sorted([pv for pv in df[plot_column].unique() if not pd.isna(pv)])

    if not plot_values:
        print(f"No valid plot values found for column '{plot_column}'. Nothing to plot.")
        return

    # print(f"Debug: plot_values to be used: {plot_values}")

    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(font_scale=2.0)
    sns.set_style("whitegrid")

    plot_kwargs_base: Dict[str, Any] = {}
    plot_func: Any  # Function placeholder
    if plot_type == "boxplot":
        plot_func = sns.boxplot
        plot_kwargs_base = dict(
            notch=False,
            showmeans=True,
            meanprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black", markersize="6"),
            medianprops=dict(color="black", label="_median_", linewidth=2),
            flierprops=dict(alpha=0.3),
        )
    elif plot_type == "barplot":
        plot_func = sns.barplot
        plot_kwargs_base = dict(errorbar="sd")
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

    final_plot_kwargs = {**plot_kwargs_base, **kwargs}

    num_plots = len(plot_values)
    fig_width, fig_height = figsize
    # Ensure a minimum height per subplot to avoid issues with very small figsize[1]
    fig_height = max(fig_height, 4)  # e.g., min 4 inches per subplot
    if vertical:
        fig_height = fig_height * num_plots
    else:
        fig_width = fig_width * num_plots

    # print(f"Debug: num_plots: {num_plots}, fig_width: {fig_width}, height_per_subplot: {height_per_subplot}, fig_height: {fig_height}")

    fig, axes_array = plt.subplots(
        nrows=num_plots if vertical else 1,
        ncols=1 if vertical else num_plots,
        figsize=(fig_width, fig_height),
        sharex=True,
        squeeze=False,  # Ensures axes_array is always 2D
    )
    axes = axes_array.flatten()  # Makes it a 1D array for easier iteration

    all_handles, all_labels = None, None
    legend_info_collected = False

    for i, value in enumerate(plot_values):
        df_metric = df[df[plot_column] == value].copy()
        current_ax = axes[i]
        current_ax.set_title(str(value).capitalize())  # Subplot title is the current 'value'

        # print(f"Debug: Processing value '{value}'. Filtered df_metric rows: {len(df_metric)}")

        if df_metric.empty or df_metric[y_axis].isna().all():  # Also check if y_axis data is all NaN
            # print(f"Debug: No data or all NaN in y_axis for value '{value}'. Plotting 'No data' text.")
            current_ax.text(
                0.5,
                0.5,
                f"No data for\n{str(value)}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=current_ax.transAxes,
            )
            current_ax.set_yticks([])
            if i == num_plots - 1:  # Last plot
                current_ax.set_xlabel(x_axis)
                current_ax.set_xticks([])  # No data, so no x-ticks
            else:  # Not the last plot (relies on sharex to hide ticks)
                current_ax.set_xlabel("")
            continue

        df_metric.sort_values(by=[x_axis, group_by], inplace=True)

        plot_func(
            x=x_axis,
            y=y_axis,
            hue=group_by,
            data=df_metric,
            palette=seaborn_palette,
            orient="v",
            ax=current_ax,
            **final_plot_kwargs,
        )

        # y_label_for_subplot = str(value) if (y_axis_label == plot_column or y_axis_label is None) else y_axis_label
        # current_ax.set_ylabel(y_label_for_subplot)
        current_ax.tick_params(axis="x", rotation=45)
        current_ax.set_xlabel("")
        current_ax.set_ylabel("")  # (ylabel='')

        # if i == num_plots - 1:  # Last plot
        #     current_ax.set_xlabel(x_axis)
        # else:  # Not the last plot
        #     current_ax.set_xlabel("")  # X-label is cleared due to sharex

        # Collect legend info from the first valid plot
        if not legend_info_collected and current_ax.get_legend() is not None:
            handles, labels = current_ax.get_legend_handles_labels()
            if handles:  # Ensure there are actual legend items
                unique_legend_items: Dict[str, Any] = {}
                for h, l_item in zip(handles, labels):  # Renamed l to l_item
                    if l_item not in unique_legend_items:
                        unique_legend_items[l_item] = h
                all_handles = list(unique_legend_items.values())
                all_labels = list(unique_legend_items.keys())
                legend_info_collected = True

        if current_ax.get_legend() is not None:
            current_ax.get_legend().remove()

    # --- Configure overall figure appearance (legend, suptitle) and layout ---
    rect_top = 1.0  # Initial top boundary for plot content area

    # if title:
    #     # title_size = float(plt.rcParams["figure.titlesize"]) * 1.2
    #     fig.suptitle(title, fontsize="xx-large", y=0.98)
    #     rect_top = 0.92  # Lower plot area to make space for suptitle

    if all_handles and all_labels:
        num_legend_cols = len(df[group_by].unique())
        legend_y_anchor = rect_top  # Place legend starting from current top edge

        fig.legend(
            all_handles,
            all_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y_anchor),  # Anchor top of legend
            title=group_by.capitalize(),
            ncol=min(num_legend_cols, 4),  # Max 5 columns for legend
            handletextpad=0.1,  # Adjust space between handle and text
            borderaxespad=0.1,
        )
        # Estimate legend height reduction factor (very approximate)
        num_legend_rows = (len(all_labels) + min(num_legend_cols, 5) - 1) // min(num_legend_cols, 5)
        legend_height_factor = 0.04 * num_legend_rows  # Roughly 4% height per legend row
        rect_top -= (legend_height_factor + 0.04)  # Reduce plot area for legend + small gap #NOTE: use + gap to move legend upwards
        # print(f"Debug: Legend height factor: {legend_height_factor}")

    # Ensure rect_top is not too low, preventing plot collapse
    rect_top = max(rect_top, 0.1)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, rect_top], pad=0.5)
    # fig.tight_layout()

    # --- Save the figure ---
    # Sanitize title and plot_column for filename
    clean_title_part = title.lower().replace(" ", "_").replace("/", "_") if title else None
    clean_plot_column_part = plot_column.lower().replace(" ", "_").replace("/", "_")

    out_file_parts = [prefix, plot_type]
    if clean_title_part:
        out_file_parts.append(clean_title_part)
    else:
        out_file_parts.append(f"combined_{clean_plot_column_part}")

    out_filename = "_".join(filter(None, out_file_parts)) + ".png"

    # print(f"Debug: Saving to {outdir / out_filename}")
    # print(f"Debug: Figure size inches: {fig.get_size_inches()}")
    # print(f"Debug: Final rect for tight_layout: {[0.03, 0.03, 0.97, rect_top]}")

    # For local debugging, you might want to see the plot:
    # plt.tight_layout()

    plt.savefig(outdir / out_filename, dpi=400)
    plt.clf()  # Clear the current figure
    plt.close(fig)  # Close the figure window/object
    # print(f"Debug: Plot saved and figure closed: {out_filename}")


def grouped_violin_plot(outdir: Path, df: pd.DataFrame, x_axis: str, y_axis: str, group_by: str, plot_column: str,
                        figsize=(12, 8), plot_values: List[str] = None, plot_type: str = "boxplot",
                        y_axis_label: str = None, title: str = None, prefix: str = "", **kwargs):
    # Create a boxplot for each model
    if plot_values is None:
        plot_values = df[plot_column].unique().tolist()
    for value in plot_values:
        df_metric = df[df[plot_column] == value]

        if df_metric.empty:
            print(f"No data available for {value}. Skipping...")
            continue

        # Order x-axis and group columns
        x_order = kwargs.get('order')
        if x_order:
            df_metric.loc[:, x_axis] = pd.Categorical(df_metric[x_axis], categories=x_order, ordered=True)
        group_labels_order = kwargs.get('hue_order')
        if group_labels_order:
            df_metric.loc[:, group_by] = pd.Categorical(df_metric[group_by], categories=group_labels_order, ordered=True)

        # Create the boxplot
        fig = go.Figure()
        unique_groups = kwargs.get("hue_order", df_metric[group_by].unique())  # Get unique categories from your group_by column

        # palette = px.colors.qualitative.Plotly
        for i_g, group_category_value in enumerate(unique_groups):
            df_specific_group = df_metric[df_metric[group_by] == group_category_value]

            if not df_specific_group.empty:
                fig.add_trace(go.Violin(
                    x=df_specific_group[x_axis],
                    y=df_specific_group[y_axis],
                    name=str(group_category_value),
                    legendgroup=str(group_category_value),
                    scalegroup=str(group_category_value),
                    # line_color=palette[2] if side == "positive" else palette[1],
                    scalemode="count",
                    line=dict(width=2),  # Set the thickness of the lines
                    meanline=dict(
                        color="black",
                        visible=True,
                        width=3  # Set the thickness of the mean line
                    )
                ))

        # min_y_value, max_y_value = float(df_specific_group[y_axis].min()), float(df_specific_group[y_axis].max())
        # min_y_value, max_y_value = min_y_value - min_y_value * 0.15, max_y_value + max_y_value * 0.1

        # plot_title_str: str = f"{title} for {value_to_plot}" if title else f"Violin Plot for {value_to_plot} grouped by {group_by}"
        y_axis_title_str: str = y_axis_label if y_axis_label != plot_column else value.capitalize()
        fig.update_traces(opacity=1.0, meanline_visible=True, jitter=0.1, points=False)

        FONT_SIZE = 24
        font = dict(size=FONT_SIZE, color="black", family="Arial")
        fig.update_layout(
            autosize=True,
            # violingap=0.0,
            # bargap=0,
            # violingroupgap=0.3,
            violinmode='group',  # 'overlay' for split view if sides are correctly set
            # title=plot_title_str,
            # xaxis_title=x_axis.replace("_", " ").capitalize(),  # Add x-axis title
            yaxis_title=y_axis_title_str,
            plot_bgcolor='white',  # White background
            yaxis_showgrid=True,  # Show y-axis grid
            yaxis_gridcolor='lightgrey',  # Set grid color
            xaxis_showgrid=False,  # Optionally hide x-axis grid
            font=font,
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",
                y=1.02,  # Position legend above the plot
                xanchor="center",  # Center legend
                x=0.5,
                font=font
            ),
            margin=dict(l=50, r=10, t=40, b=40),
            xaxis=dict(
                showline=True,  # Show x-axis line
                linewidth=1,  # Line width
                linecolor='black',  # Line color
                mirror=True,  # Makes the line appear on top as well if against plot_bgcolor
                title_font=font,  # X-axis title font size
                # showticklabels=False  # Hide x-axis tick labels
            ),
            yaxis=dict(
                # range=[min_y_value, 1],
                showline=True,  # Show y-axis line
                linewidth=1,
                linecolor='black',
                mirror=True,
                dtick=0.1,
                title_font=font  # Y-axis title font size
            ),
            # showlegend=False
        )

        if x_order:
            fig.update_xaxes(categoryorder='array', categoryarray=list(x_order))

        # Show the plot (optional, as we are saving it)
        # fig.show()

        # SAVE the figure
        outdir.mkdir(parents=True, exist_ok=True)
        # Construct filename
        filename_prefix = prefix if prefix else ""
        filename_title_part = f"_{title.lower().replace(' ', '_')}" if title else ""
        out_file_name = f"{filename_prefix}violin_{value}{filename_title_part}.png"

        output_path = outdir / out_file_name

        print(f"Saving plot to {output_path}")
        try:
            # Ensure figsize is used for image export if desired, though plotly uses width/height in layout
            fig.write_image(output_path, width=figsize[0] * 96, height=figsize[1] * 96, scale=3)
            print(f"Plot saved successfully to {output_path}")
        except Exception as e:
            print(f"Error saving plot: {e}. Make sure 'kaleido' package is installed.")


def grouped_box_plot(outdir: Path, df: pd.DataFrame, x_axis: str, y_axis: str, group_by: str, plot_column: str,
                     figsize=(12, 8), plot_values: List[str] = None, plot_type: str = "boxplot",
                     y_axis_label: str = None, title: str = None, prefix: str = "", **kwargs):
    # Create a boxplot for each model
    if plot_values is None:
        plot_values = df[plot_column].unique().tolist()
    for value in plot_values:
        df_metric = df[df[plot_column] == value]

        if df_metric.empty:
            print(f"No data available for {value}. Skipping...")
            continue

        # Order x-axis and group columns
        x_order = kwargs.get('order')
        if x_order:
            df_metric.loc[:, x_axis] = pd.Categorical(df_metric[x_axis], categories=x_order, ordered=True)
        group_labels_order = kwargs.get('hue_order')
        if group_labels_order:
            df_metric.loc[:, group_by] = pd.Categorical(df_metric[group_by], categories=group_labels_order, ordered=True)

        # Create the boxplot
        fig = go.Figure()
        unique_groups = kwargs.get("hue_order", df_metric[group_by].unique())  # Get unique categories from your group_by column

        for i_g, group_category_value in enumerate(unique_groups):
            df_specific_group = df_metric[df_metric[group_by] == group_category_value]

            if not df_specific_group.empty:
                fig.add_trace(go.Box(
                    x=df_specific_group[x_axis],
                    y=df_specific_group[y_axis],
                    name=str(group_category_value),
                    legendgroup=str(group_category_value),
                    line=dict(color=seaborn_palette[i_g], width=3)
                ))

        y_axis_title_str: str = y_axis_label if y_axis_label != plot_column else value.capitalize()
        fig.update_traces(opacity=1.0, boxmean=True, boxpoints=False, orientation="v")

        FONT_SIZE = 26
        font = dict(size=FONT_SIZE, color="black", family="Arial")
        fig.update_layout(
            boxgap=0.5,
            boxgroupgap=0.0,
            boxmode='group',  # 'overlay' for split view if sides are correctly set
            # title=plot_title_str,
            # xaxis_title=x_axis.replace("_", " ").capitalize(),  # Add x-axis title
            yaxis_title=y_axis_title_str,
            plot_bgcolor='white',  # White background
            xaxis_showgrid=False,  # Show y-axis grid
            yaxis_gridcolor='lightgrey',  # Set grid color
            yaxis_showgrid=True,  # Optionally hide x-axis grid
            font=font,
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",
                y=1.02,  # Position legend above the plot
                xanchor="center",  # Center legend
                x=0.5,
                font=font
            ),
            margin=dict(l=50, r=10, t=40, b=40),
            xaxis=dict(
                showline=True,  # Show x-axis line
                linewidth=1,  # Line width
                linecolor='black',  # Line color
                mirror=True,  # Makes the line appear on top as well if against plot_bgcolor
                title_font=font,  # X-axis title font size
                # showticklabels=False  # Hide x-axis tick labels
            ),
            yaxis=dict(
                # range=[min_y_value, 1],
                showline=True,  # Show y-axis line
                linewidth=1,
                linecolor='black',
                mirror=True,
                title_font=font  # Y-axis title font size
            ),
            # showlegend=False
        )

        if x_order:
            fig.update_xaxes(categoryorder='array', categoryarray=list(x_order))

        # Show the plot (optional, as we are saving it)
        # fig.show()

        # SAVE the figure
        outdir.mkdir(parents=True, exist_ok=True)

        # Construct filename
        out_file = f"{prefix}{plot_type}_{value}"
        if title is not None:
            out_file += f"_{title.lower().replace(' ', '_')}"
        out_file += ".png"

        output_path = outdir / out_file

        print(f"Saving plot to {output_path}")
        try:
            # Ensure figsize is used for image export if desired, though plotly uses width/height in layout
            fig.write_image(output_path, width=figsize[0] * 96, height=figsize[1] * 96, scale=3)
            print(f"Plot saved successfully to {output_path}")
        except Exception as e:
            print(f"Error saving plot: {e}. Make sure 'kaleido' package is installed.")


def split_violin_plot(outdir: Path, df: pd.DataFrame, x_axis: str, y_axis: str, group_by: str, plot_column: str,
                      figsize=(12, 8), plot_values: List[str] = None, plot_type: str = "boxplot",
                      y_axis_label: str = None, title: str = None, prefix: str = "", **kwargs):
    # Create a boxplot for each model
    if plot_values is None:
        plot_values = df[plot_column].unique().tolist()
    for value in plot_values:
        df_metric = df[df[plot_column] == value]

        if df_metric.empty:
            print(f"No data available for {value}. Skipping...")
            continue

        # Order x-axis and group columns
        x_order = kwargs.get('order')
        if x_order:
            df_metric.loc[:, x_axis] = pd.Categorical(df_metric[x_axis], categories=x_order, ordered=True)
        group_labels_order = kwargs.get('hue_order')
        if group_labels_order:
            df_metric.loc[:, group_by] = pd.Categorical(df_metric[group_by], categories=group_labels_order, ordered=True)
        # df_metric = df_metric.sort_values(by=[x_axis])

        # Create the boxplot
        fig = go.Figure()
        unique_groups = kwargs.get("hue_order", df_metric[group_by].unique())  # Get unique categories from your group_by column

        for i_g, group_category_value in enumerate(unique_groups):
            df_specific_group = df_metric[df_metric[group_by] == group_category_value]  # .sort_values(by=x_axis)
            if i_g < len(unique_groups) // 2:
                side = "negative"
            else:
                side = "positive"
            if not df_specific_group.empty:
                fig.add_trace(go.Violin(
                    x=df_specific_group[x_axis],
                    y=df_specific_group[y_axis],
                    name=str(group_category_value),  # Name for the legend entry (e.g., the specific group)
                    legendgroup=str(group_category_value),  # Groups related traces in the legend
                    side=side,
                    scalegroup=str(group_category_value),
                    line_color=seaborn_palette[4] if side == "positive" else seaborn_palette[5],
                    line=dict(width=2),  # Set the thickness of the lines
                    meanline=dict(
                        color="black",
                        visible=True,
                        width=3  # Set the thickness of the mean line
                    )
                ))

        min_y_value, max_y_value = float(df_specific_group[y_axis].min()), float(df_specific_group[y_axis].max())
        min_y_value, max_y_value = min_y_value - min_y_value * 0.2, max_y_value + max_y_value * 0.2

        # plot_title_str: str = f"{title} for {value_to_plot}" if title else f"Violin Plot for {value_to_plot} grouped by {group_by}"
        y_axis_title_str: str = y_axis_label if y_axis_label != plot_column else value.capitalize()
        fig.update_traces(opacity=1.0, meanline_visible=True, jitter=0.1, scalemode='width', points=False, width=1.0)

        FONT_SIZE = 30
        font = dict(size=FONT_SIZE, color="black", family="Arial")
        fig.update_layout(
            violingap=0.0,
            violingroupgap=0.0,
            violinmode='overlay',  # 'overlay' for split view if sides are correctly set
            # title=plot_title_str,
            # xaxis_title=x_axis.replace("_", " ").capitalize(),  # Add x-axis title
            yaxis_title=y_axis_title_str,
            plot_bgcolor='white',  # White background
            yaxis_showgrid=True,  # Show y-axis grid
            yaxis_gridcolor='lightgrey',  # Set grid color
            xaxis_showgrid=False,  # Optionally hide x-axis grid
            font=font,
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",
                y=1.02,  # Position legend above the plot
                xanchor="center",  # Center legend
                x=0.5,
                font=font
            ),
            margin=dict(l=50, r=10, t=10, b=10),
            xaxis=dict(
                showline=True,  # Show x-axis line
                linewidth=1,  # Line width
                linecolor='black',  # Line color
                mirror=True,  # Makes the line appear on top as well if against plot_bgcolor
                title_font=font,  # X-axis title font size
                showticklabels=kwargs.get("showticklabels", True)  # Hide x-axis tick labels
            ),
            yaxis=dict(
                # range=[min_y_value, max_y_value],
                showline=True,  # Show y-axis line
                linewidth=1,
                linecolor='black',
                mirror=True,
                dtick=0.10,
                title_font=font  # Y-axis title font size
            ),
            showlegend=kwargs.get("showlegend", True)  # Show legend
        )

        if x_order:
            fig.update_xaxes(categoryorder='array', categoryarray=list(x_order), tickangle=30)

        # Show the plot (optional, as we are saving it)
        # fig.show()

        # SAVE the figure
        outdir.mkdir(parents=True, exist_ok=True)
        # Construct filename
        out_file = f"{prefix}{plot_type}_{value}"
        if title is not None:
            out_file += f"_{title.lower().replace(' ', '_')}"
        out_file += ".png"

        output_path = outdir / out_file

        print(f"Saving plot to {output_path}")
        try:
            # Ensure figsize is used for image export if desired, though plotly uses width/height in layout
            fig.write_image(output_path, width=figsize[0] * 96, height=figsize[1] * 96, scale=3)
            print(f"Plot saved successfully to {output_path}")
        except Exception as e:
            print(f"Error saving plot: {e}. Make sure 'kaleido' package is installed.")
