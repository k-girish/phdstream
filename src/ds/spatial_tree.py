import os
from collections import deque
from typing import Union

import pandas as pd
from geopandas import GeoSeries
from matplotlib import pyplot as plt
from networkx import MultiDiGraph
import osmnx as ox
from shapely import Point
import numpy as np
from shapely.prepared import prep

from src.ds import dp_counter as dpc
from src.utils.spatial import spatial_division_into_two_with_divide_dim, \
    spatial_division_into_n_children, random_points_in_geometry, convert_points_to_df, SingleOrMultiPolygon, \
    SingleOrMultiLineString


class SpatialTreeNode:
    def __init__(
            self,
            geometry: Union[SingleOrMultiPolygon, SingleOrMultiLineString, MultiDiGraph],
            parent,
            depth=0,
            metric=0.0,
            true_metric=0.0,
            divide_dimension=0,
            has_counter=False,
            counter_type='simple',
            counter_eps=None,
            is_leaf=True,
            node_id='id_0'
    ):
        self.geometry = geometry
        self.metric = metric
        self.parent = parent
        self.depth = depth
        self.true_metric = true_metric
        self.divide_dimension = divide_dimension
        self.is_leaf = is_leaf
        self.node_id = node_id

        self.children = []

        self.height = 1

        # Initialize a counter if needed
        self.has_counter = has_counter
        if self.has_counter:
            self.initialize_counter(counter_type, counter_eps)

        # Prepared geometry for easy query
        if not isinstance(self.geometry, MultiDiGraph):
            if isinstance(self.geometry, SingleOrMultiLineString):
                self.buffered_geometry = self.geometry.buffer(1e-6)
            else:
                self.buffered_geometry = self.geometry

            self.prepared_geometry = prep(self.buffered_geometry)
        else:
            self.sorted_edge_list = None

    # --------------------- When geometry is a graph ---------------------
    def can_split(self, n_children=2):
        if self.sorted_edge_list is not None:
            return len(self.sorted_edge_list) >= n_children
        else:
            return len(ox.graph_to_gdfs(self.geometry)[1]) >= n_children

    def get_sorted_edge_list(self):
        vdf, edf = ox.graph_to_gdfs(self.geometry)
        temp = {}
        for k, r in edf.iterrows():
            u = k[0]
            v = k[1]
            temp[k] = {
                'ux': vdf.loc[u, 'x'],
                'uy': vdf.loc[u, 'y'],
                'vx': vdf.loc[v, 'x'],
                'vy': vdf.loc[v, 'y']
            }

        temp = pd.DataFrame(temp).T
        if self.depth % 2 == 0:
            self.sorted_edge_list = temp.sort_values(by=['ux', 'uy', 'vx', 'vy']).index.values
        else:
            self.sorted_edge_list = temp.sort_values(by=['vy', 'vx', 'uy', 'ux']).index.values
        return self.sorted_edge_list

    def create_children_in_graph(self, n_children=2):
        assert n_children == 2
        if self.can_split(n_children=n_children):
            self.get_sorted_edge_list()

            splits = np.array_split(self.sorted_edge_list, n_children)
            for c_idx, es in enumerate(splits):
                temp_new_child = SpatialTreeNode(
                    geometry=self.geometry.edge_subgraph(es).copy(),
                    parent=self,
                    depth=self.depth + 1,
                    has_counter=self.has_counter,
                    counter_type=self.counter_type if self.has_counter else None,
                    counter_eps=self.counter_eps if self.has_counter else None,
                    node_id=self.node_id + str(c_idx)
                )
                temp_new_child.sorted_edge_list = es
                self.children.append(temp_new_child)

        return self.children

    def find_count_and_points_on_graph_edges(self, parent_df: pd.DataFrame, nearest_edges=None):
        assert parent_df is not None
        if (len(parent_df) == 0) or (self.parent is None):
            return parent_df

        # Find the nearest edge on the parent graph, if this node is root, it is the parent as well
        if nearest_edges is None:
            nearest_edges = np.array(
                ox.distance.nearest_edges(self.parent.geometry, parent_df['x'].values, parent_df['y'].values)
            )

        # Check whether this edge is in the subgraph "g" or not
        assert len(nearest_edges) != 0

        edge_in_g_bool_arr = np.apply_along_axis(lambda x: self.geometry.has_edge(x[0], x[1], x[2]), axis=1,
                                                 arr=nearest_edges)
        return parent_df[edge_in_g_bool_arr]

    def random_sample_on_graph_edges(self, n):
        ans = []

        vdf, edf = ox.graph_to_gdfs(self.geometry)
        e_list = edf.index.values
        n_edges = len(e_list)

        edge_choices = np.random.choice(n_edges, size=n, replace=True)
        edge_choices = e_list[edge_choices]
        edge_choices, edge_counts = np.unique(edge_choices, return_counts=True)

        for edge, c in zip(edge_choices, edge_counts):
            geo = edf.loc[edge]['geometry']
            points = geo.interpolate(np.random.uniform(low=0, high=1, size=c), normalized=True)
            points = [list(point.coords) for point in points]
            ans.append(points)

        if len(ans) > 0:
            ans = np.concatenate(ans).reshape(-1, 2)

        return ans

    # --------------------- When geometry is a graph ---------------------

    def update_metric(self, calculated, true=-1):
        self.metric = calculated
        self.true_metric = true

    def initialize_counter(self, counter_type, eps, initial_offset=0.0):
        self.has_counter = True
        self.counter_type = counter_type
        self.counter_eps = eps
        self.counter = dpc.get_counter_based_on_type(
            self.counter_type, eps=self.counter_eps, initial_offset=initial_offset
        )
        self.ancestor_counting = 0
        self.descendant_counting = 0
        self.count_history = []

    def count(self, n, time_in_algorithm=None):
        assert self.has_counter
        self.counter.count(n)
        self.update_metric_based_on_counter(update_type='self', counted_val=n, counting_time=time_in_algorithm)

    def verify_ac_dc_by_iteration(self):
        # AC
        ac = 0
        temp = self.parent
        p_level = 2.0
        while temp is not None:
            ac += (temp.counter.value / p_level)
            p_level *= 2.0
            temp = temp.parent

        assert round(self.ancestor_counting, 2) == round(ac, 2)

        # DC
        dc = 0
        traverse = deque(self.children)
        while len(traverse) > 0:
            temp = traverse.popleft()
            dc += temp.counter.value
            for cnode in temp.children:
                traverse.append(cnode)
        assert round(self.descendant_counting, 2) == round(dc, 2)

        # Metric
        assert self.metric == (
                self.counter.value + self.ancestor_counting + self.descendant_counting
        )

    def update_ancestor_counting(self, new_ac):
        self.ancestor_counting = new_ac
        self.update_metric_based_on_counter(update_type='ancestor')

    def update_descendant_counting(self, new_dc):
        self.descendant_counting = new_dc
        self.update_metric_based_on_counter(update_type='descendant')

    def update_metric_based_on_counter(
            self,
            update_type='self',
            counted_val=None, counting_time=None,
            rounding_and_non_neg=False
    ):
        self.metric = self.ancestor_counting + self.counter.value + self.descendant_counting
        if rounding_and_non_neg:
            self.metric = max(0, round(self.metric))

        # Store history of all counter things
        # self.count_history.append({
        #     'id': self.node_id,
        #     'type': update_type,
        #     'm': self.metric,
        #     'cv': self.counter.value,
        #     'counted_val': counted_val,
        #     'time': counting_time,
        #     'ac': self.ancestor_counting,
        #     'dc': self.descendant_counting
        # })

    def create_children(self, divide_into_n_geometries=2, mark_parent_non_leaf=True, save_geo_compare_plot=False):
        if isinstance(self.geometry, MultiDiGraph):
            self.create_children_in_graph(n_children=divide_into_n_geometries)
        else:
            if divide_into_n_geometries == 2:
                divide_dimension = (self.divide_dimension + 1) % 2
                child_geoms = spatial_division_into_two_with_divide_dim(
                    self.geometry, divide_dim=divide_dimension
                )
            else:
                divide_dimension = None
                child_geoms = spatial_division_into_n_children(
                    self.geometry, n_children=divide_into_n_geometries
                )

            for c_idx, child_geometry in enumerate(child_geoms):
                self.add_child(
                    SpatialTreeNode(
                        geometry=child_geometry,
                        parent=self,
                        depth=self.depth + 1,
                        divide_dimension=divide_dimension,
                        has_counter=self.has_counter,
                        counter_type=self.counter_type if self.has_counter else None,
                        counter_eps=self.counter_eps if self.has_counter else None,
                        node_id=self.node_id + str(c_idx)
                    )
                )

        # Increasing height till the root
        self.height += 1
        self.increment_height_of_ancestors_if_needed(self.parent)

        # Mark parent as non-leaf ?
        self.is_leaf = not mark_parent_non_leaf

        if save_geo_compare_plot:
            f, ax = plt.subplots(1, divide_into_n_geometries + 1, figsize=(15, 10))
            all_geoms = [self.geometry] + child_geoms
            for ax_idx, geom in enumerate(all_geoms):
                gs = GeoSeries([geom])
                gs.plot(ax=ax[ax_idx], facecolor="none", edgecolor='black', lw=0.7, linestyle='dashed', alpha=0.3)
                ax[ax_idx].axis('off')

            plot_path = os.path.join(
                "/Users/girish/EverythingElse/experiment_results",
                f"children_plot_depth_{self.depth}.png"
            )
            f.savefig(plot_path)
            plt.close(f)

    def increment_height_of_ancestors_if_needed(self, root):
        if root is not None:
            curr_height = root.height
            new_height = curr_height
            for c in root.children:
                new_height = max(new_height, c.height)
            if new_height > curr_height:
                root.height = new_height
                self.increment_height_of_ancestors_if_needed(root.parent)

    def add_child(self, child):
        self.children.append(child)

    def filter_data_in_geometry(self, parent_df: pd.DataFrame, nearest_edges=None):
        if isinstance(self.geometry, MultiDiGraph):
            return self.find_count_and_points_on_graph_edges(parent_df, nearest_edges=nearest_edges)
        else:
            return parent_df[self.prepared_geometry.contains(parent_df['points'])]

    def generate_random_points_inside(self, n, cols=['x', 'y']):
        if isinstance(self.geometry, MultiDiGraph):
            points = self.random_sample_on_graph_edges(n)
            df = pd.DataFrame(points, columns=cols)
            df['points'] = df[['x', 'y']].apply(lambda x: Point((x['x'], x['y'])), axis=1)
        else:
            points = random_points_in_geometry(self.geometry, n)
            df = convert_points_to_df(points, cols=cols)
            df['points'] = points
        return df

    def is_very_small(self, tol=1e-6):
        if isinstance(self.geometry, MultiDiGraph):
            return not self.can_split(n_children=2)
        elif isinstance(self.geometry, SingleOrMultiLineString):
            return self.geometry.length < tol
        else:
            return self.geometry.area < tol

    def evaluate_queries(self, queries: GeoSeries):
        # Prefill result 0 for all queries
        query_result = pd.Series([0] * len(queries), index=queries.index)

        # Find intersecting queries
        intersecting_query_indices: pd.Index = queries[queries.intersects(self.buffered_geometry)].index
        if len(intersecting_query_indices) == 0:
            return query_result

        # If query contains the geometry, return full metric
        containing_query_indices = queries[intersecting_query_indices].contains(self.buffered_geometry)
        containing_query_indices: pd.Index = queries[intersecting_query_indices][containing_query_indices].index
        query_result[containing_query_indices] = self.metric

        # If query partially intersects with the geometry
        partially_containing_query_indices = intersecting_query_indices.difference(containing_query_indices)
        if len(partially_containing_query_indices) == 0:
            return query_result
        p_queries = queries[partially_containing_query_indices].intersection(self.buffered_geometry)

        # Call for children, if they exist and the current node is not leaf
        if (not self.is_leaf) and self.children:
            for cnode in self.children:
                child_q_result = cnode.evaluate_queries(p_queries)
                query_result = query_result.add(child_q_result, fill_value=0)
        # return metric based on area
        else:
            query_result[partially_containing_query_indices] = (
                                                                       p_queries.area / self.buffered_geometry.area
                                                               ) * self.metric

        return query_result

    # TODO: this is not needed when we have much more info in node counting histories
    def node_counting_times(self) -> pd.DataFrame:
        ans = []
        traverse = deque([self])
        while len(traverse) > 0:
            temp = traverse.popleft()
            ans.append(temp.counter.curr_time)
            for cnode in temp.children:
                traverse.append(cnode)
        return pd.DataFrame(ans)

    def node_counting_histories(self) -> pd.DataFrame:
        ans = []
        traverse = deque([self])
        while len(traverse) > 0:
            temp = traverse.popleft()
            ans.extend(temp.count_history)
            for cnode in temp.children:
                traverse.append(cnode)
        return pd.DataFrame(ans)


# -------------------------------------- Class definition ends here ----------------------------
# -------------------------------------- Class definition ends here ----------------------------
# -------------------------------------- Class definition ends here ----------------------------

def find_leaves_of_a_tree(root):
    leaves = []
    traverse = deque([root])
    while (len(traverse) > 0):
        temp = traverse.popleft()
        if len(temp.children) == 0:
            leaves.append(temp)
        else:
            for c in temp.children:
                traverse.append(c)
    return leaves


def re_populate_tree_for_given_total_points(root: SpatialTreeNode, n_total: int):
    root.metric = n_total

    if not root.children:
        return

    if n_total > 0:
        tot = 0.0
        for cn in root.children:
            tot += cn.metric

        if tot > 0:
            c_metrics = np.array([c.metric for c in root.children])
            c_props = c_metrics.astype(float) / c_metrics.sum()
            c_metrics = np.round(c_props * n_total)

            for c_idx, cn in enumerate(root.children):
                cn.metric = c_metrics[c_idx]

            # Find some indices where we randomly remove each remaining metric
            remain = int(n_total - c_metrics.sum())
            if remain != 0:
                change_indices = np.random.choice(len(c_metrics), size=abs(remain))
                if remain > 0:
                    for c_idx in change_indices:
                        root.children[c_idx].metric += 1
                else:
                    for c_idx in change_indices:
                        root.children[c_idx].metric -= 1
        else:
            divide_metric_equally_to_all_child(root)

        # Call for each child
        for cn in root.children:
            re_populate_tree_for_given_total_points(cn, cn.metric)

    else:
        for cn in root.children:
            re_populate_tree_for_given_total_points(cn, 0)


def update_ancestor_count_for_children(root: SpatialTreeNode, proportional_split=False):
    if root.children:
        if root.has_counter:
            if proportional_split:
                c_props = np.array([cn.counter.value + cn.descendant_counting for cn in root.children])
                c_props = np.maximum(c_props, 0)
                if c_props.sum() == 0:
                    c_props = np.ones(len(root.children), dtype=float) / len(root.children)
                else:
                    c_props = c_props / c_props.sum()
            else:
                c_props = np.ones(len(root.children), dtype=float) / len(root.children)

            new_ancestor_count_for_children = float(root.counter.value + root.ancestor_counting) * c_props
            for cn, new_ancestor_count_for_cn in zip(root.children, new_ancestor_count_for_children):
                cn.update_ancestor_counting(new_ancestor_count_for_cn)


def divide_metric_equally_to_all_child(root: SpatialTreeNode, proportional_split=False):
    if root.children:
        if root.has_counter:
            update_ancestor_count_for_children(root, proportional_split=proportional_split)
        else:
            new_count_for_child = float(root.metric) / len(root.children)
            for cn in root.children:
                cn.metric = new_count_for_child


def setup_counter_for_existing_tree(root: SpatialTreeNode, counter_type, eps):
    if root.children:
        root.initialize_counter(counter_type, eps)
        root.descendant_counting = root.metric
        for cn in root.children:
            setup_counter_for_existing_tree(cn, counter_type, eps)
    else:
        root.initialize_counter(counter_type, eps, initial_offset=root.metric)
