import os
from typing import List

import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from shapely import Point
from tqdm import tqdm

from src.ds.spatial_tree import SpatialTreeNode, re_populate_tree_for_given_total_points
from src.utils import dict_with_prefix, find_save_path, save_results
import src.utils.time as time_util
from src.utils.plot import generate_scatter_plot_compare_figure
from src.utils.spatial import get_polygon_from_x_y_range, get_polygon_boundary_for_data, get_x_y_range_for_geometry, \
    query_error_by_type, count_geo_series_geometry_contains_geo_series_points


def analyze_privtree(data: pd.DataFrame, holdout_data: pd.DataFrame, hyper_param_combinations, iterations_per_config,
                     samplings_per_iter,
                     save_prefix):
    results = []
    save_path = None

    for hyper_param_dict in tqdm(hyper_param_combinations, desc='hyper params'):
        epsilon = hyper_param_dict["epsilon"]
        privacy_budget_ratio = hyper_param_dict["privacy_budget_ratio"]

        for _ in tqdm(range(iterations_per_config), desc='experiment iteration'):
            priv_tree_params = PrivTree.Params(
                data=data,
                epsilon=epsilon,
                theta=0,
                max_depth=20,
                privacy_budget_ratio=privacy_budget_ratio)

            priv_tree_obj = PrivTree(data, priv_tree_params)

            # Fit and save time taken
            fit_result_dict = priv_tree_obj.fit()

            for _ in tqdm(range(samplings_per_iter), desc='sampling iteration'):
                scatter_plot_save_path = find_save_path(sub_folder=os.path.join("pmm", "figures"),
                                                        prefix=save_prefix,
                                                        extension="png")
                priv_tree_obj.params.fig_save_path = scatter_plot_save_path
                synth_data, result_dict = priv_tree_obj.generate(plot_scatter=False)
                result_dict["plot_file"] = scatter_plot_save_path

                # Adding existing data to the result
                result_dict.update(fit_result_dict)

                # Hyper params
                result_dict.update(dict_with_prefix(priv_tree_params.__dict__, "params"))

                # Parameters about data creation
                result_dict["n"] = data.shape[0]
                result_dict["p"] = 2

                # Save synthetic data
                syn_df_save_path = save_results(synth_data,
                                                sub_folder=os.path.join("pmm_comp",
                                                                        "syn_df"),
                                                prefix="privtree_syn_df")
                result_dict["syn_df_path"] = syn_df_save_path

                # Algo name
                result_dict["algo"] = "privtree"

                # Saving results
                results.append(result_dict)
                results_df = pd.DataFrame(results)
                save_path = save_results(results_df, sub_folder="pmm", prefix=save_prefix, path=save_path)


def calculate_privtree_privacy_params_and_add_to_obj(obj):
    assert obj.epsilon is not None
    assert obj.privacy_budget_ratio is not None
    assert obj.fanout_beta is not None

    # Effective privacy budget
    obj.fit_epsilon = obj.epsilon * obj.privacy_budget_ratio
    obj.tree_epsilon = obj.epsilon - obj.fit_epsilon

    # Choice of lambda
    if obj.tree_epsilon == 0.0:
        obj.lambda_tree_noise_scale = 0.0
    else:
        obj.lambda_tree_noise_scale = ((2 * obj.fanout_beta - 1) / (obj.fanout_beta - 1)) / obj.tree_epsilon

    # Delta
    obj.gamma = np.log(obj.fanout_beta)
    obj.delta = obj.lambda_tree_noise_scale * obj.gamma


class PrivTree:
    class Params:

        def __init__(
                self,
                data,
                epsilon,
                theta,  # count threshold to split a node of the tree
                geometry=None,
                re_populate_with_n=None,
                max_depth=20,
                fanout_beta=2.0,
                privacy_budget_ratio=0.5,
                x_range=None,
                y_range=None,
                fig_save_path="../privtree_plot.png",
                show_grid_in_plot=True,
                plot_subplots_in_landscape=True,
                data_col_names=['x', 'y']
        ):
            # Update data related metrics in the params
            self.n = float(data.shape[0])
            self.re_populate_with_n = re_populate_with_n

            self.epsilon = epsilon
            self.theta = theta
            self.geometry = geometry
            self.fanout_beta = fanout_beta
            self.max_depth = max_depth
            self.privacy_budget_ratio = privacy_budget_ratio
            self.x_range = x_range
            self.y_range = y_range

            self.fig_save_path = fig_save_path
            self.show_grid_in_plot = show_grid_in_plot
            self.plot_subplots_in_landscape = plot_subplots_in_landscape

            self.data_col_names = data_col_names

            # Calculating the privacy params
            calculate_privtree_privacy_params_and_add_to_obj(self)

        def __str__(self):
            d = self.__dict__.copy()
            d.pop("x_col", None)
            d.pop("y_col", None)
            return d.__str__()

    def __init__(self, data: pd.DataFrame, params: Params, query_df: gpd.GeoDataFrame = None):
        self.params = params
        self.data = data.copy(deep=True)
        self.query_df = query_df
        self.true_query_answers = None

        if 'points' not in self.data:
            self.data['points'] = self.data[['x', 'y']].apply(lambda x: Point((x['x'], x['y'])), axis=1)

        self.leaf_nodes: List[SpatialTreeNode] = []
        self.leaf_node_geometries = []

        self.partition_plotting_counter = 0

    def __process_leaf_node__(self, root: SpatialTreeNode):
        self.leaf_nodes.append(root)
        self.leaf_node_geometries.append(root.geometry)

    def __generate_tree_recursive__(self, root: SpatialTreeNode, parent_data: pd.DataFrame, curr_depth=0):
        filtered_data = root.filter_data_in_geometry(parent_data)
        count_of_node = filtered_data.shape[0]
        biased_count_of_node = count_of_node - curr_depth * self.params.delta
        biased_count_of_node = max(biased_count_of_node, self.params.theta - self.params.delta)
        noisy_biased_count_of_node = biased_count_of_node + np.random.laplace(loc=0.0,
                                                                              scale=self.params.lambda_tree_noise_scale)

        if (noisy_biased_count_of_node > self.params.theta) and (not root.is_very_small()) and (
                root.depth < self.params.max_depth):
            root.create_children(divide_into_n_geometries=self.params.fanout_beta)
            children_synth_metric_sum = 0
            for child_node in root.children:
                child_node_synthetic_count = self.__generate_tree_recursive__(child_node, filtered_data, curr_depth + 1)
                children_synth_metric_sum += child_node_synthetic_count
            root.update_metric(children_synth_metric_sum, count_of_node)
            return children_synth_metric_sum

        else:
            if self.params.fit_epsilon == 0.0:
                noise = 0.0
            else:
                noise = np.random.laplace(loc=0.0, scale=1.0 / float(self.params.fit_epsilon))
            sm = max(round(count_of_node + noise), 0)
            root.update_metric(sm, count_of_node)
            self.__process_leaf_node__(root)
            return sm

    def generate_tree(self):
        if self.params.geometry is None:
            if (self.params.x_range is not None) and (self.params.y_range is not None):
                self.params.geometry = get_polygon_from_x_y_range(self.params.x_range, self.params.y_range)
            else:
                self.params.geometry = get_polygon_boundary_for_data(self.data)
        else:
            self.params.x_range, self.params.y_range = get_x_y_range_for_geometry(self.params.geometry)

        self.tree_root = SpatialTreeNode(geometry=self.params.geometry, parent=None)
        self.__generate_tree_recursive__(self.tree_root, self.data)

    def fit(self):
        _tk_fit_time = time_util.start_time_record("fit_time")

        self.generate_tree()
        self.fit_result_dict = {}

        if self.params.re_populate_with_n is not None:
            re_populate_tree_for_given_total_points(self.tree_root, self.params.re_populate_with_n)

        # Finding the result of queries
        if self.query_df is not None:
            if self.true_query_answers is None:
                self.true_query_answers = count_geo_series_geometry_contains_geo_series_points(
                    self.query_df.geometry, self.data['points']
                )

            # We don't need these tree based queries
            # fit_tree_query_results: pd.Series = self.tree_root.evaluate_queries(queries=self.query_df.geometry)
            # query_error_dict = query_error_by_type(
            #     self.true_query_answers, fit_tree_query_results, self.query_df, len(self.data), prefix='tree_query'
            # )
            # self.fit_result_dict.update(query_error_dict)

        fit_duration = time_util.end_time_record(_tk_fit_time)

        self.fit_result_dict['fit_duration'] = fit_duration
        return self.fit_result_dict

    def generate(self, plot_scatter=True):
        _tk_generate_time = time_util.start_time_record("gen_time")

        synth_data = pd.DataFrame(data=[], columns=self.params.data_col_names)
        for node in self.leaf_nodes:
            sampled_df = node.generate_random_points_inside(
                int(node.metric),
                cols=self.params.data_col_names
            )
            synth_data = pd.concat([synth_data, sampled_df], axis=0)

        duration = time_util.end_time_record(_tk_generate_time)

        result = {
            "synth_size": len(synth_data),
            "sample_duration": duration,
            "n_leaf_nodes": len(self.leaf_nodes)
        }

        # Calculating the query error for synth data
        if (self.query_df is not None) and (self.true_query_answers is not None):
            synth_data_query_answers = count_geo_series_geometry_contains_geo_series_points(
                self.query_df.geometry, synth_data['points']
            )
            query_error_dict = query_error_by_type(
                self.true_query_answers, synth_data_query_answers, self.query_df, len(self.data), prefix='query'
            )
            result.update(query_error_dict)

        fig = generate_scatter_plot_compare_figure(self, synth_data, show_figure=plot_scatter, other_info_dict=result,
                                                   show_grid=self.params.show_grid_in_plot,
                                                   landscape_subplots=self.params.plot_subplots_in_landscape)
        fig.savefig(self.params.fig_save_path)
        plt.close(fig)

        return synth_data, result

    def fit_generate(self, plot_scatter=True):
        fit_result = self.fit()
        synth_data, sample_result = self.generate(plot_scatter=plot_scatter)
        sample_result.update(fit_result)
        return synth_data, sample_result
