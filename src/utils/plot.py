import itertools
import random

import pandas as pd
from geopandas import GeoSeries
from matplotlib import pyplot as plt
from networkx import MultiDiGraph

from src.utils import create_query_for_given_cols_and_values


def plot_constant_val_on_axes(ax, constant_val, constant_val_title=None):
    ax.axhline(y=constant_val, color='black', linestyle='--', )
    xmin, _ = ax.get_xlim()
    y_step = plt.yticks()[0][1] - plt.yticks()[0][0]
    x_step = plt.xticks()[0][1] - plt.xticks()[0][0]
    if constant_val_title:
        ax.text(
            xmin + x_step / 2,
            constant_val - y_step / 2,
            constant_val_title,
            {'color': 'black', 'size': 'small', 'rotation': 'horizontal'}
        )


def plot_df_x_vs_y_with_given_legend_cols(df_original: pd.DataFrame, x_col, y_col, legend_cols, title_suffix=None,
                                          constant_val=None,
                                          constant_val_title=None,
                                          title=None,
                                          x_label=None,
                                          y_label=None,
                                          legend_labels=None,
                                          continuing_fig=None,
                                          continuing_ax=None):
    print(f"Plotting {x_col} vs {y_col}")
    all_cols = [x_col, y_col]
    all_cols = all_cols + legend_cols
    df = df_original[all_cols]

    vals = [df[col].unique() for col in legend_cols]

    # For successive call to this function plotting on a continuing previous instance
    if continuing_fig is None:
        fig, ax = plt.subplots(figsize=(15, 5))
    else:
        fig = continuing_fig
        ax = continuing_ax

    for val_comb in itertools.product(*vals):
        print(f"Legends = {legend_cols}")
        print(f"Val comb = {val_comb}")
        # Find x and y values for the specific legend val combination
        query = create_query_for_given_cols_and_values(legend_cols, val_comb)
        temp = df.query(query)
        print(f"After query: {temp.shape[0]}")

        if len(temp) > 0:
            temp = temp[[x_col, y_col]]

            plot_mean_and_std_on_axes(temp, x_col, y_col, ax, query)

    if constant_val is not None:
        plot_constant_val_on_axes(ax, constant_val, constant_val_title=constant_val_title)

    if legend_labels is not None:
        ax.legend(legend_labels, loc="upper left")
    else:
        ax.legend(loc="upper left")
    ax.set_xlabel(x_label if x_label is not None else x_col)
    ax.set_ylabel(y_label if y_label is not None else y_col)
    ax.set_title(title if title is not None else f"{x_col} vs {y_col}, {title_suffix}")
    plt.show()
    return fig, ax


def plot_df_x_vs_y_with_given_legend_cols_on_given_axes(
        ax,
        df_original: pd.DataFrame, x_col, y_col, legend_cols,
        constant_col=None,
        constant_val_title=None,
        legend_labels=None
):
    print(f"Plotting {x_col} vs {y_col}")
    all_cols = [x_col, y_col]
    all_cols = all_cols + legend_cols
    if constant_col is not None:
        all_cols.append(constant_col)
    df = df_original[all_cols]

    vals = [df[col].unique() for col in legend_cols]

    for val_comb in itertools.product(*vals):
        print(f"Legends = {legend_cols}")
        print(f"Val comb = {val_comb}")
        # Find x and y values for the specific legend val combination
        query = create_query_for_given_cols_and_values(legend_cols, val_comb)
        temp = df.query(query)
        print(f"After query: {temp.shape[0]}")

        if len(temp) > 0:
            temp = temp[[x_col, y_col]]

            plot_mean_and_std_on_axes(temp, x_col, y_col, ax, query)

    if constant_col is not None:
        constant_val = df[constant_col].values[0]
        c = (random.random(), random.random(), random.random())
        ax.axhline(y=constant_val, color=c, linestyle='--', )
        xmin, _ = ax.get_xlim()
        y_step = plt.yticks()[0][1] - plt.yticks()[0][0]
        x_step = plt.xticks()[0][1] - plt.xticks()[0][0]
        ax.text(xmin + x_step / 2, constant_val - y_step / 2, constant_val_title,
                {'color': c, 'size': 'small', 'rotation': 'horizontal'})

    if legend_labels is not None:
        ax.legend(legend_labels, loc="upper left")
    else:
        ax.legend(loc="upper left")


# Note: s[0] is the mean, i.e. the position vector for x-coordinate and not std
# Note: s[0] is the mean, i.e. the position vector for x-coordinate and not std
# Note: s[0] is the mean, i.e. the position vector for x-coordinate and not std
def plot_on_axis_with_given_mean_and_std(ax, m, s, label, marker='o'):
    ax.plot(m[0], m[1], f"--{marker}", label=label)
    ax.fill_between(s[0], m[1] - s[1], m[1] + s[1], alpha=0.2)


def plot_mean_and_std_on_axes(df, x_col, y_col, ax, label, marker='o'):
    # Calculate the mean y for each x and plot
    temp_mean = df.groupby(by=[x_col])[y_col].mean().reset_index()
    # Calculate the std of y for each x and plot
    temp_std = df.groupby(by=[x_col])[y_col].std().reset_index()
    plot_on_axis_with_given_mean_and_std(
        ax,
        (temp_mean[x_col], temp_mean[y_col]),
        (temp_std[x_col], temp_std[y_col]),
        label=label,
        marker=marker
    )


def plot_synth_data_with_partitions(data, algo_obj, sc_color, sc_alpha, sc_size, show_grid=True):
    f, ax = plt.subplots()
    ax.scatter(data['x'], data['y'], alpha=sc_alpha, c=sc_color, s=sc_size)
    ax.axis('off')
    if show_grid:
        if hasattr(algo_obj, 'leaf_node_geometries') and (
                not isinstance(algo_obj.leaf_node_geometries[0], MultiDiGraph)):
            gs = GeoSeries(algo_obj.leaf_node_geometries)
            gs.plot(ax=ax, facecolor="none", edgecolor='black', lw=0.5, linestyle='dashed', alpha=0.3)

    if hasattr(algo_obj.params, 'x_range') and hasattr(algo_obj.params, 'y_range'):
        ax.set_xlim(algo_obj.params.x_range[0], algo_obj.params.x_range[1])
        ax.set_ylim(algo_obj.params.y_range[0], algo_obj.params.y_range[1])

    f.subplots_adjust(hspace=0.0, wspace=0.0)

    return f


def true_synth_generate_scatter_plot_compare_figure(obj, synth_data, transparency=0.2,
                                                    scatter_size=0.6,
                                                    show_grid=True,
                                                    base_save_path='.'):
    # True
    fig_1 = plot_synth_data_with_partitions(obj.data, obj, sc_color="C0", sc_alpha=transparency, sc_size=scatter_size,
                                            show_grid=show_grid)
    fig_1.savefig(f"{base_save_path}_true.png")
    plt.close(fig_1)

    # Synth
    fig_2 = plot_synth_data_with_partitions(synth_data, obj, sc_color="C1", sc_alpha=transparency, sc_size=scatter_size,
                                            show_grid=show_grid)
    fig_2.savefig(f"{base_save_path}_synth.png")
    plt.close(fig_2)


def generate_scatter_plot_compare_figure(obj, synth_data, show_figure=True, transparency=0.2, other_info_dict={},
                                         scatter_size=0.6,
                                         show_grid=True,
                                         include_info=True,
                                         landscape_subplots=False):
    color1 = "C0"
    color2 = "C1"
    if landscape_subplots:
        f, ax = plt.subplots(2, 1, figsize=(15, 10))
    else:
        f, ax = plt.subplots(1, 2, figsize=(15, 10))

    # First Subplot
    ax[0].scatter(obj.data['x'], obj.data['y'], alpha=transparency, label='True', c=color1,
                  s=scatter_size)
    ax[0].legend(loc="upper left")
    ax[0].title.set_text("True data on scatter plot")

    # Second Subplot
    ax[1].scatter(synth_data['x'], synth_data['y'], alpha=transparency, label='Synth', c=color2,
                  s=scatter_size)
    ax[1].legend(loc="upper left")
    ax[0].title.set_text("Synth data on scatter plot")

    if show_grid:
        if hasattr(obj, 'leaf_node_geometries') and (not isinstance(obj.leaf_node_geometries[0], MultiDiGraph)):
            gs = GeoSeries(obj.leaf_node_geometries)
            # This is not helpful to visualize line strings
            # if isinstance(obj.leaf_node_geometries[0], SingleOrMultiLineString):
            #     gs.plot(ax=ax[0], facecolor="none", edgecolor='red', lw=0.7, linestyle='dashed', alpha=0.3)
            #     gs.plot(ax=ax[1], facecolor="none", edgecolor='red', lw=0.7, linestyle='dashed', alpha=0.3)
            # else:
            gs.plot(ax=ax[0], facecolor="none", edgecolor='black', lw=0.7, linestyle='dashed', alpha=0.3)
            gs.plot(ax=ax[1], facecolor="none", edgecolor='black', lw=0.7, linestyle='dashed', alpha=0.3)
            ax[0].axis('off')
            ax[1].axis('off')

        elif hasattr(obj.leaf_nodes[0], 'boundary') and (obj.leaf_nodes[0].boundary.d == 2):
            for itr, node in enumerate(obj.leaf_nodes):
                rect0 = node.boundary.get_rectangular_patch_for_plot()
                rect1 = node.boundary.get_rectangular_patch_for_plot()

                # Adding rectangle patches to both
                ax[0].add_patch(rect0)
                ax[1].add_patch(rect1)

    if hasattr(obj.params, 'x_range') and hasattr(obj.params, 'y_range'):
        ax[0].set_xlim(obj.params.x_range[0], obj.params.x_range[1])
        ax[0].set_ylim(obj.params.y_range[0], obj.params.y_range[1])
        ax[1].set_xlim(obj.params.x_range[0], obj.params.x_range[1])
        ax[1].set_ylim(obj.params.y_range[0], obj.params.y_range[1])

    if include_info:
        pr_copy = obj.params.__dict__.copy()
        if "noise_levels" in pr_copy:
            del pr_copy['noise_levels']

        if hasattr(obj, "fit_result_dict"):
            plt.figtext(0.5, 0.01,
                        str(pr_copy) + obj.fit_result_dict.__str__() + other_info_dict.__str__(),
                        wrap=True, horizontalalignment='center')
        else:
            plt.figtext(0.5, 0.01,
                        str(pr_copy) + obj.fit_results[-1].__str__() + other_info_dict.__str__(),
                        wrap=True, horizontalalignment='center')

    if show_figure:
        plt.show()
    return f
