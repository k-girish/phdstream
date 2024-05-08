import os.path
import time
from multiprocessing import Pool
from typing import List, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoSeries
from networkx import MultiDiGraph
from shapely import MultiPolygon, Polygon, Point
import osmnx as ox
from tqdm import tqdm

import src.utils.time as time_utils
from src.algo.privtree import PrivTree
from src.ds.spatial_tree import SpatialTreeNode, find_leaves_of_a_tree, \
    divide_metric_equally_to_all_child, re_populate_tree_for_given_total_points, \
    setup_counter_for_existing_tree, update_ancestor_count_for_children
from src.utils import dict_with_prefix
from src.utils.plot import true_synth_generate_scatter_plot_compare_figure
from src.utils.spatial import get_x_y_range_for_geometry, count_geo_series_geometry_contains_geo_series_points, \
    query_error_by_type


def biased_coin_toss(prob):
    return np.random.choice(2, p=[1 - prob, prob]) == 1


def __priv_tree_recursive_helper__(root: SpatialTreeNode, data: pd.DataFrame, leaves: List[SpatialTreeNode],
                                   partition_noise_scale, count_noise_scale, delta, theta,
                                   max_depth, prob_leaf_ct=1):
    filtered_data = root.filter_data_in_geometry(data)
    count_of_node = filtered_data.shape[0]
    biased_count_of_node = count_of_node + root.metric - root.depth * delta
    biased_count_of_node = max(biased_count_of_node, theta - delta)
    noisy_count_of_node = biased_count_of_node + np.random.laplace(loc=0.0, scale=partition_noise_scale)

    # Along with other condition, check if we get 1 from a biased coin toss
    if ((root.depth < max_depth) and (noisy_count_of_node > theta)) \
            and (biased_coin_toss(1.0 / (2 ** min(root.depth, 4)))):

        root.create_children(divide_into_n_geometries=2)
        root.children[0].metric = root.metric // 2
        root.children[1].metric = root.metric - root.children[0].metric

        child_metric_0 = __priv_tree_recursive_helper__(
            root.children[0], filtered_data, leaves,
            partition_noise_scale, count_noise_scale, delta, theta, max_depth,
            prob_leaf_ct=prob_leaf_ct
        )
        child_metric_1 = __priv_tree_recursive_helper__(
            root.children[1], filtered_data, leaves,
            partition_noise_scale, count_noise_scale, delta, theta, max_depth,
            prob_leaf_ct=prob_leaf_ct
        )
        root.metric = child_metric_0 + child_metric_1

    else:
        noise = np.random.laplace(loc=0.0, scale=count_noise_scale)
        root.metric = max(round(root.metric + count_of_node + noise), 0)
        leaves.append(root)

    return root.metric


def _process_leaf_nodes_in_arr_(node_arr, new_data, leaves: List[SpatialTreeNode],
                                partition_noise_scale, count_noise_scale, delta, theta,
                                max_depth, prob_leaf_ct=1):
    for node in node_arr:
        __priv_tree_recursive_helper__(node, new_data, leaves,
                                       partition_noise_scale, count_noise_scale, delta, theta,
                                       max_depth=max_depth, prob_leaf_ct=prob_leaf_ct)
    return leaves


class PHDStream:
    class Params:

        METHOD_PHDSTREAM = 'PHDSTREAM'
        METHOD_PRIV_TREE_WITH_DATA_SO_FAR = 'PRIV_TREE_WITH_DATA_SO_FAR'
        METHOD_PRIV_TREE_WITH_CURRENT_DATA = 'PRIV_TREE_WITH_CURRENT_DATA'
        METHOD_PRIV_TREE_INIT_AND_COUNTING = 'PRIV_TREE_INIT_AND_COUNTING'

        def __init__(
                self,
                epsilon,
                geometry: Union[Polygon, MultiPolygon, GeoSeries],
                method=None,
                counter_type=None,
                infer_time_from_batch_sizes=False,
                total_time=None,
                initialization_time=0,
                batch_size=None,
                initialization_data_ratio=0,
                leaf_node_count_threshold=0.0,
                max_depth=20,
                fanout_beta=2.0,
                initial_depth=10,
                data_dimension=2,
                change_sensitivity=1.0,
                n_synth_data_per_time=1,
                exp_id=None,
                fig_save_folder="../temp",
                plot_subplots_in_landscape=True,
                syn_df_save_folder="../temp_syn_df",
                data_col_names=['x', 'y'],
                x_range=None,
                y_range=None,
                save_generated_synth_data=False,
                re_populate_based_on_true_data_count=False,
                disable_tqdm=True,
                proportional_split_ancestor_count=False
        ):
            # Input epsilon the desired epsilon
            self.epsilon_input = epsilon
            self.epsilon = epsilon / float(change_sensitivity)

            self.method = method
            self.counter_type = counter_type
            if self.counter_type is not None:
                assert (
                        (self.method == self.METHOD_PHDSTREAM)
                        or (self.method == self.METHOD_PRIV_TREE_INIT_AND_COUNTING)
                )
                self.use_counters = True
            else:
                self.use_counters = False

            self.infer_time_from_batch_sizes = infer_time_from_batch_sizes
            self.total_time = total_time
            self.initialization_time = initialization_time
            self.batch_size = batch_size
            self.initialization_data_ratio = initialization_data_ratio

            # Batch and time based assertions
            if self.infer_time_from_batch_sizes:
                assert self.batch_size is not None
                assert self.total_time is None
                assert self.initialization_time == 0
            else:
                assert self.total_time is not None
                assert self.batch_size is None
                assert self.initialization_data_ratio == 0

            # Sensitivity can be calculated if given limit data by user
            self.change_sensitivity = change_sensitivity

            self.leaf_node_count_threshold = leaf_node_count_threshold
            self.max_depth = max_depth
            self.fanout_beta = fanout_beta
            self.initial_depth = initial_depth
            self.data_dimension = data_dimension
            self.n_synth_data_per_time = n_synth_data_per_time
            self.data_col_names = data_col_names
            self.x_range = x_range
            self.y_range = y_range

            # Processing the geometry and x, y, bounds
            self.geometry: Union[Polygon, MultiPolygon, GeoSeries] = geometry
            if (self.geometry is not None) and ((self.x_range is None) or (self.y_range is None)):
                self.x_range, self.y_range = get_x_y_range_for_geometry(self.geometry)
            assert self.x_range is not None
            assert self.y_range is not None

            self.exp_id = exp_id
            self.syn_df_save_folder = syn_df_save_folder
            self.fig_save_folder = fig_save_folder
            self.plot_subplots_in_landscape = plot_subplots_in_landscape

            self.save_generated_synth_data = save_generated_synth_data
            self.re_populate_based_on_true_data_count = re_populate_based_on_true_data_count
            self.proportional_split_ancestor_count = proportional_split_ancestor_count

            self.disable_tqdm = disable_tqdm

            # Updating the parameters required by privtree
            self.get_privtree_params()

        def get_privtree_params(self):

            self.theta = self.leaf_node_count_threshold
            # Choice of lambda
            if self.epsilon == 0.0:
                self.pt_partition_noise_scale = 0.0
                self.pt_count_noise_scale = 0.0
            else:
                self.pt_partition_noise_scale = ((2 * self.fanout_beta - 1) / (self.fanout_beta - 1))
                self.pt_partition_noise_scale = self.pt_partition_noise_scale / (self.epsilon / 2.0)
                self.pt_count_noise_scale = 2.0 / self.epsilon

            # Delta
            gamma = np.log(self.fanout_beta)
            self.delta = self.pt_partition_noise_scale * gamma

        def as_dict(self):
            d = self.__dict__.copy()
            d.pop("tree_root", None)
            d.pop("geometry", None)
            d.pop("data_col_names", None)
            d.pop("per_level_epsilon", None)
            return d

        def __str__(self):
            return self.as_dict().__str__()

        # Params class ends here
        # Params class ends here
        # Params class ends here
        # Params class ends here

    def __priv_tree_with_motion_recursive_helper__(self, root: SpatialTreeNode, data: pd.DataFrame):

        filtered_data = root.filter_data_in_geometry(data)
        if 'move' in filtered_data.columns:
            new_count_of_node = filtered_data['move'].values.sum()
        else:
            new_count_of_node = len(filtered_data)

        biased_count_of_node = new_count_of_node + root.metric - (root.depth * self.params.delta)
        biased_count_of_node = max(biased_count_of_node, self.params.theta - self.params.delta)
        noisy_count_of_node = biased_count_of_node + np.random.laplace(loc=0.0,
                                                                       scale=self.params.pt_partition_noise_scale)

        # Split the node
        if (
                (root.depth < self.params.max_depth) and
                (noisy_count_of_node > self.params.theta)
        ):
            if not root.children:
                root.create_children(divide_into_n_geometries=self.params.fanout_beta)
                divide_metric_equally_to_all_child(root)
        # Merge the node
        else:
            if root.children:
                root.children = []

        if root.children:
            c_sum = 0
            for cn in root.children:
                self.__priv_tree_with_motion_recursive_helper__(cn, filtered_data)
                c_sum += cn.metric

            # Maintains consistency
            root.metric = c_sum

        else:
            noise = np.random.laplace(loc=0.0, scale=self.params.pt_count_noise_scale)
            new_noisy_count = max(round(new_count_of_node + root.metric + noise), 0)
            root.metric = new_noisy_count
            self.leaf_nodes.append(root)
            self.leaf_node_geometries.append(root.geometry)

    def __priv_tree_with_motion_and_counter_recursive_helper__(self, root: SpatialTreeNode, data: pd.DataFrame):

        filtered_data = root.filter_data_in_geometry(data)
        if 'move' in filtered_data.columns:
            true_batch_count_of_node = filtered_data['move'].values.sum()
        else:
            true_batch_count_of_node = len(filtered_data)

        # Assert all calculation is correct for ac and dc
        # root.verify_ac_dc_by_iteration()

        biased_count_of_node = true_batch_count_of_node + root.metric - (root.depth * self.params.delta)
        biased_count_of_node = max(biased_count_of_node, self.params.theta - self.params.delta)
        noisy_count_of_node = biased_count_of_node + np.random.laplace(loc=0.0,
                                                                       scale=self.params.pt_partition_noise_scale)

        # Split the node
        if (
                (root.depth < self.params.max_depth) and
                (noisy_count_of_node > self.params.theta)
        ):
            root.is_leaf = False

            # If the node is not split so far, first create children
            if not root.children:
                root.create_children(divide_into_n_geometries=self.params.fanout_beta)

            # Update the ancestor count for the children to account for any changes in the past
            update_ancestor_count_for_children(root, proportional_split=self.params.proportional_split_ancestor_count)

            # Finding the descendant count
            new_descendant_counting = 0
            for cn in root.children:
                self.__priv_tree_with_motion_and_counter_recursive_helper__(cn, filtered_data)
                new_descendant_counting += (cn.counter.value + cn.descendant_counting)

            # Maintains consistency
            root.update_descendant_counting(new_descendant_counting)

        # Make this node leaf
        else:
            root.is_leaf = True
            root.count(true_batch_count_of_node, self.time_so_far)
            self.leaf_nodes.append(root)
            self.leaf_node_geometries.append(root.geometry)

    def phdstream(self, new_data: pd.DataFrame):

        # New leaves will be stored here
        self.leaf_nodes = []
        self.leaf_node_geometries = []

        if self.params.use_counters:
            self.__priv_tree_with_motion_and_counter_recursive_helper__(
                self.tree_root, new_data
            )
        else:
            self.__priv_tree_with_motion_recursive_helper__(
                self.tree_root, new_data
            )

        if self.params.re_populate_based_on_true_data_count and (self.tree_root.metric > len(self.data_with_motion)):
            re_populate_tree_for_given_total_points(self.tree_root, len(self.data_with_motion))

    def only_counting_at_leaves(self, data: pd.DataFrame):
        # Ideally We should only pass data of parent in call to filter_data_in_geometry
        # Alternatively, for multidigraph we can do a hack
        if isinstance(self.tree_root.geometry, MultiDiGraph):
            nearest_edges = np.array(
                ox.distance.nearest_edges(self.tree_root.geometry, data['x'].values, data['y'].values)
            )
        else:
            nearest_edges = None

        for ln in self.leaf_nodes:
            filtered_data = ln.filter_data_in_geometry(data, nearest_edges)
            if 'move' in filtered_data.columns:
                true_batch_count_of_node = filtered_data['move'].values.sum()
            else:
                true_batch_count_of_node = len(filtered_data)

            ln.count(true_batch_count_of_node, self.time_so_far)

    def priv_tree_on_data_so_far(self, calc_metrics):
        if calc_metrics:
            priv_tree_params = PrivTree.Params(
                data=self.data,
                epsilon=self.params.epsilon,
                theta=0.0,
                max_depth=self.params.max_depth,
                fanout_beta=self.params.fanout_beta,
                show_grid_in_plot=True,
                x_range=self.params.x_range,
                y_range=self.params.y_range,
                geometry=self.params.geometry,
                fig_save_path=os.path.join(
                    self.params.fig_save_folder, f"priv_tree_plot_time_{self.time_so_far}_idx_0.png"
                ),
                plot_subplots_in_landscape=self.params.plot_subplots_in_landscape
            )
            priv_tree_obj = PrivTree(self.data, priv_tree_params, query_df=self.query_df)

            priv_tree_synth_data, priv_tree_result = priv_tree_obj.fit_generate(plot_scatter=False)

            if self.params.save_generated_synth_data:
                priv_tree_synth_data.to_csv(
                    os.path.join(self.params.syn_df_save_folder, f"syn_df_time_{self.time_so_far}_idx_0.csv"),
                    index=False
                )

            self.tree_root = priv_tree_obj.tree_root

            # self.fit_synth_results.append(priv_tree_result)

            # Find leaf nodes using the Spatial tree
            self.leaf_nodes: List[SpatialTreeNode] = find_leaves_of_a_tree(self.tree_root)
            self.leaf_node_geometries = [ln.geometry for ln in self.leaf_nodes]

    def priv_tree_on_current_data(self, new_data: pd.DataFrame, calc_metrics):
        if calc_metrics:
            if 'move' in new_data.columns:
                temp_group = new_data.groupby(by=['x', 'y']).sum(numeric_only=True)
                temp_group = temp_group[temp_group['move'] > 0]
                curr_data = temp_group.reset_index()
                curr_data['points'] = curr_data[['x', 'y']].apply(lambda p: Point((p['x'], p['y'])), axis=1)
            else:
                curr_data = new_data

            priv_tree_params = PrivTree.Params(
                data=curr_data,
                epsilon=self.params.epsilon,
                theta=0.0,
                re_populate_with_n=len(self.data),
                max_depth=self.params.max_depth,
                fanout_beta=self.params.fanout_beta,
                show_grid_in_plot=True,
                x_range=self.params.x_range,
                y_range=self.params.y_range,
                geometry=self.params.geometry,
                fig_save_path=os.path.join(
                    self.params.fig_save_folder, f"priv_tree_plot_time_{self.time_so_far}_idx_0.png"
                ),
                plot_subplots_in_landscape=self.params.plot_subplots_in_landscape
            )
            priv_tree_obj = PrivTree(curr_data, priv_tree_params, query_df=self.query_df)

            priv_tree_synth_data, _ = priv_tree_obj.fit_generate(plot_scatter=False)

            if self.params.save_generated_synth_data:
                priv_tree_synth_data.to_csv(
                    os.path.join(self.params.syn_df_save_folder, f"syn_df_time_{self.time_so_far}_idx_0.csv"),
                    index=False
                )

            self.tree_root = priv_tree_obj.tree_root

            # Find leaf nodes using the Spatial tree
            self.leaf_nodes: List[SpatialTreeNode] = find_leaves_of_a_tree(self.tree_root)
            self.leaf_node_geometries = [ln.geometry for ln in self.leaf_nodes]

    def concat_data_accounting_for_motion(self, new_data_points_as_df: pd.DataFrame):
        if self.data_with_motion is None:
            self.data_with_motion = new_data_points_as_df
        else:
            self.data_with_motion = pd.concat([self.data_with_motion, new_data_points_as_df], axis=0)

        if 'move' in new_data_points_as_df.columns:
            temp_df = self.data_with_motion[['x', 'y', 'move']]
            temp_group = temp_df.groupby(by=['x', 'y']).sum(numeric_only=True)
            temp_group = temp_group[temp_group['move'] != 0]
            assert np.all(temp_group['move'].values > 0)
            self.data = temp_group.reset_index()
            self.data['points'] = self.data[['x', 'y']].apply(lambda p: Point((p['x'], p['y'])), axis=1)


        else:
            self.data = self.data_with_motion

    def __init__(self, params: Params, df: pd.DataFrame = None, query_df: gpd.GeoDataFrame = None):
        self.params = params
        self.query_df = query_df

        # Non-parameterized initializations
        self.time_so_far = 0
        self.all_data = df
        self.data_with_motion = None
        self.data = None
        self.synth_results = []
        self.fit_synth_results = []
        self.true_query_results_at_this_time = None

        # Initializing partition tree
        if (
                self.params.infer_time_from_batch_sizes and (self.params.initialization_data_ratio > 0)
        ) or (
                (not self.params.infer_time_from_batch_sizes) and (self.params.initialization_time != 0)
        ):
            assert df is not None

            if self.params.infer_time_from_batch_sizes:
                self.params.initialization_data_length = int(self.params.initialization_data_ratio * len(df))
                initialization_data = df.iloc[:self.params.initialization_data_length, :]
            else:
                initialization_data = df[df['time'] <= self.params.initialization_time]
                self.params.initialization_data_length = len(initialization_data)

            self.concat_data_accounting_for_motion(initialization_data)
            priv_tree_params = PrivTree.Params(
                data=self.data,
                epsilon=self.params.epsilon,
                theta=0.0,
                max_depth=self.params.max_depth,
                fanout_beta=self.params.fanout_beta,
                show_grid_in_plot=True,
                x_range=self.params.x_range,
                y_range=self.params.y_range,
                geometry=self.params.geometry,
                fig_save_path=os.path.join(self.params.fig_save_folder, "privtree_initial.png"),
                plot_subplots_in_landscape=self.params.plot_subplots_in_landscape
            )
            priv_tree_obj = PrivTree(self.data, priv_tree_params, query_df=self.query_df)

            # Fit and save time taken
            priv_tree_init_fit_result = priv_tree_obj.fit()

            init_synth_data, priv_tree_init_result = priv_tree_obj.generate(plot_scatter=False)
            priv_tree_init_result.update(priv_tree_init_fit_result)

            if self.params.save_generated_synth_data:
                init_synth_data.to_csv(os.path.join(self.params.syn_df_save_folder, 'syn_df_init.csv'), index=False)

            self.priv_tree_init_result = priv_tree_init_result
            self.init_synth_data = init_synth_data
            self.fit_synth_results.append(priv_tree_init_result)

            self.tree_root = priv_tree_obj.tree_root

            # Tree root
            if self.params.use_counters:
                if self.params.method == self.params.METHOD_PHDSTREAM:
                    counter_eps_for_initialized_tree = 1.0 / self.params.pt_count_noise_scale
                elif self.params.method == self.params.METHOD_PRIV_TREE_INIT_AND_COUNTING:
                    counter_eps_for_initialized_tree = self.params.epsilon
                else:
                    raise Exception(f"Cannot find counter eps for method {self.params.method}")
                setup_counter_for_existing_tree(
                    self.tree_root,
                    counter_type=self.params.counter_type,
                    eps=counter_eps_for_initialized_tree
                )
                self.params.counter_class = type(self.tree_root.counter)
        else:
            self.tree_root = SpatialTreeNode(
                geometry=self.params.geometry, parent=None,
                has_counter=self.params.use_counters,
                counter_type=self.params.counter_type,
                counter_eps=1.0 / self.params.pt_count_noise_scale if self.params.use_counters else None
            )
            self.params.initialization_data_length = 0

        # Find leaf nodes using the Spatial tree
        self.leaf_nodes: List[SpatialTreeNode] = find_leaves_of_a_tree(self.tree_root)
        self.leaf_node_geometries = [ln.geometry for ln in self.leaf_nodes]

    def fit(self, new_data_points_as_df: pd.DataFrame, calc_metrics=False):
        self.time_so_far += 1

        fit_key = time_utils.start_time_record(f'fit{self.time_so_far}')

        # Add data to the dataset seen so far
        self.concat_data_accounting_for_motion(new_data_points_as_df)

        if (self.params.method is None) or self.params.method == self.params.METHOD_PHDSTREAM:
            self.phdstream(new_data_points_as_df)

        elif self.params.method == self.params.METHOD_PRIV_TREE_WITH_DATA_SO_FAR:
            self.priv_tree_on_data_so_far(calc_metrics)

        elif self.params.method == self.params.METHOD_PRIV_TREE_WITH_CURRENT_DATA:
            self.priv_tree_on_current_data(new_data_points_as_df, calc_metrics)

        elif self.params.method == self.params.METHOD_PRIV_TREE_INIT_AND_COUNTING:
            self.only_counting_at_leaves(new_data_points_as_df)

        else:
            raise Exception(f"Unknown value {self.params.method} for method")

        duration = time_utils.end_time_record(fit_key)

        result = {
            "n": len(self.data),
            "n_with_motion": len(self.data_with_motion),
            "time": self.time_so_far,
            "fit_duration": duration,
            "n_leaf_nodes": len(self.leaf_nodes)
        }

        self.true_query_results_at_this_time = None
        if calc_metrics and (self.query_df is not None):
            # True query results
            self.true_query_results_at_this_time = count_geo_series_geometry_contains_geo_series_points(
                self.query_df.geometry, self.data['points']
            )

            # We don't need these tree query results
            # fit_tree_query_results: pd.Series = self.tree_root.evaluate_queries(queries=self.query_df.geometry)
            # query_error_dict = query_error_by_type(
            #     self.true_query_results_at_this_time, fit_tree_query_results, self.query_df, len(self.data),
            #     prefix='tree_query'
            # )
            # result.update(query_error_dict)

        return result

    def generate(self, plot_scatter=True, calc_metrics=False, generation_index=0):
        initial_time = time.time()

        synth_data = None
        for node in self.leaf_nodes:
            sampled_df = node.generate_random_points_inside(
                max(round(node.metric), 0),
                self.params.data_col_names
            )
            if synth_data is None:
                synth_data = sampled_df
            else:
                synth_data = pd.concat([synth_data, sampled_df], axis=0)

        duration = round(time.time() - initial_time, 2)

        result = {
            "n": len(self.data),
            "n_with_motion": len(self.data_with_motion),
            "synth_size": len(synth_data),
            "sample_duration": duration,
            "time": self.time_so_far,
            "synth_data_idx": generation_index
        }

        if calc_metrics:
            # Range Queries
            if (self.query_df is not None) and (self.true_query_results_at_this_time is not None):
                # Evaluate queries on the synthetic data generated
                query_result_series: pd.Series = count_geo_series_geometry_contains_geo_series_points(
                    self.query_df.geometry, synth_data['points']
                )

                # Finding error metrics between true and synth data query values
                query_error_dict = query_error_by_type(
                    self.true_query_results_at_this_time, query_result_series, self.query_df, len(self.data),
                    prefix='query'
                )
                result.update(query_error_dict)

                # Saving the query results to file
                syn_df_query_result_file = os.path.join(
                    self.params.syn_df_save_folder,
                    f"syn_df_query_result_{self.time_so_far}_idx_{generation_index}.csv"
                )
                if self.params.save_generated_synth_data:
                    query_result_series.to_csv(syn_df_query_result_file)
                    result['syn_df_query_result_path'] = syn_df_query_result_file

        if plot_scatter:
            # With grid
            plot_path = os.path.join(
                self.params.fig_save_folder,
                f"plot_time_{self.time_so_far}_idx_{generation_index}"
            )
            true_synth_generate_scatter_plot_compare_figure(
                self, synth_data, show_grid=True, base_save_path=plot_path
            )
            result['plot_path'] = plot_path

            # Without grid
            plot_path = os.path.join(
                self.params.fig_save_folder,
                f"no_grid_plot_time_{self.time_so_far}_idx_{generation_index}"
            )

            true_synth_generate_scatter_plot_compare_figure(
                self, synth_data, show_grid=False, base_save_path=plot_path
            )
            result['no_gird_plot_path'] = plot_path

        self.synth_results.append(result)

        # Saving the synthetic data generated so far
        synth_save_file = os.path.join(
            self.params.syn_df_save_folder,
            f"syn_df_time_{self.time_so_far}_idx_{generation_index}.csv"
        )
        if self.params.save_generated_synth_data:
            result['synth_df_path'] = synth_save_file
            synth_data.to_csv(synth_save_file)

        return synth_data, result

    def fit_generate(self, new_data_points: pd.DataFrame, plot_scatter=True, calc_metrics=False):
        # Fit and get results
        fit_synth_result = self.fit(
            new_data_points_as_df=new_data_points, calc_metrics=calc_metrics
        )

        synth_data = None

        if calc_metrics or plot_scatter:
            synth_results = []

            # Call generate function for each sampling iteration
            for idx in range(self.params.n_synth_data_per_time):
                synth_data, sample_result = self.generate(
                    plot_scatter=plot_scatter, calc_metrics=calc_metrics,
                    generation_index=idx
                )
                synth_results.append(sample_result)

            # Find average metrics for the sampling
            synth_results_df = pd.DataFrame(synth_results)

            if calc_metrics:
                # Get mean of all calculate metrics across different samples of synthetic data generated
                fit_synth_result['synth_size'] = synth_results_df['synth_size'].mean()
                for qs_type in ['small', 'medium', 'large']:
                    for temp_e_type in ['avg_err', 'max_err', 'rel_err', 'norm_2_err']:
                        col = f"query_{qs_type}_{temp_e_type}"
                        if col in synth_results_df.columns:
                            fit_synth_result[col] = synth_results_df[col].mean()

            fit_synth_result["sample_duration"] = synth_results_df['sample_duration'].mean()

            if self.params.save_generated_synth_data:
                synth_results_file = os.path.join(self.params.syn_df_save_folder,
                                                  f"syn_result_time_{self.time_so_far}.csv")
                synth_results_df.to_csv(synth_results_file, index=False)
                fit_synth_result["synth_results_file"] = synth_results_file

        # Appending the result to object
        self.fit_synth_results.append(fit_synth_result)

        return synth_data, fit_synth_result

    def fit_generate_iterative_for_given_df(
            self,
            plot_steps=None,
            metric_steps=None
    ):
        overall_process_tk = time_utils.start_time_record("overall")

        if self.params.infer_time_from_batch_sizes:
            n_batches = len(self.all_data) // self.params.batch_size
            self.params.total_time = n_batches
        else:
            n_batches = self.params.total_time - self.params.initialization_time

        # Plot and metric steps initialization
        if plot_steps is None:
            plot_steps = max(1, n_batches // 10)
        elif plot_steps == 0:
            plot_steps = n_batches

        if metric_steps is None:
            metric_steps = max(1, n_batches // 10)

        # Iterate taking elements of a time at once
        for idx in tqdm(range(1, n_batches + 1), disable=self.params.disable_tqdm,
                        desc=f"time iter for {self.params.exp_id}"):

            # Get data for the current time index
            if self.params.infer_time_from_batch_sizes:
                low_index = self.params.initialization_data_length + ((idx - 1) * self.params.batch_size)
                if idx == n_batches:
                    new_df = self.all_data.iloc[low_index:, :]
                else:
                    high_index = low_index + self.params.batch_size
                    new_df = self.all_data.iloc[low_index:high_index, :]
            else:
                curr_time = self.params.initialization_time + idx
                new_df = self.all_data[self.all_data['time'] == curr_time]

            _, _ = self.fit_generate(
                new_data_points=new_df,
                plot_scatter=((idx == n_batches) or (idx % plot_steps == 0)),
                calc_metrics=((idx == n_batches) or (idx % metric_steps == 0))
            )

            # Save the per time results so far
            fit_synth_result_df = pd.DataFrame(self.fit_synth_results)
            fit_synth_result_df.to_csv(
                os.path.join(self.params.syn_df_save_folder, f'results_with_time.csv'),
                index=False
            )

        # Add more details to the last fit result
        last_result = self.fit_synth_results[-1]
        last_result.update(dict_with_prefix(self.params.as_dict(), "params"))

        # Find the w loss for init data
        if hasattr(self, 'priv_tree_init_result'):
            last_result.update(dict_with_prefix(self.priv_tree_init_result, 'init'))

            if (self.query_df is not None) and (self.true_query_results_at_this_time is not None):
                # Evaluate queries on the synthetic data generated
                query_result_series: pd.Series = count_geo_series_geometry_contains_geo_series_points(
                    self.query_df.geometry, self.init_synth_data['points']
                )
                query_result_series = query_result_series * (len(self.data) / float(len(self.init_synth_data)))

                # Finding error metrics between true and synth data query values
                query_error_dict = query_error_by_type(
                    self.true_query_results_at_this_time, query_result_series, self.query_df, len(self.data),
                    prefix='init_data_at_final_time_query'
                )
                last_result.update(query_error_dict)

        overall_duration = time_utils.end_time_record(overall_process_tk)
        last_result['overall_duration'] = overall_duration

        if self.params.use_counters:
            # Node counting times
            counting_time_df = self.tree_root.node_counting_times()
            counting_time_df.to_csv(os.path.join(self.params.syn_df_save_folder, f'leaf_counting_times.csv'),
                                    index=False)

            # Node counting histories
            # counting_histories_df = self.tree_root.node_counting_histories()
            # print("Saving counting history: ", self.params.syn_df_save_folder)
            # temp_tk = time_utils.start_time_record()
            # counting_histories_df.to_pickle(os.path.join(self.params.syn_df_save_folder, f'counting_histories.p'))
            # time_utils.end_time_record(temp_tk, print_duration=True, extra_msg="Counting history save time")

        return last_result
