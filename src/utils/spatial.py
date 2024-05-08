import os.path
from typing import Union, List

import geodatasets
import pandas as pd
from geopandas import GeoSeries
import geopandas as gpd
from networkx import MultiDiGraph
import osmnx as ox
from shapely import Polygon, MultiPolygon, Point, unary_union, LineString, MultiLineString, GeometryCollection
import numpy as np
from tqdm import tqdm

from src.utils import find_difference_errors_in_two_arrays, dict_with_prefix

SingleOrMultiPolygon = Union[Polygon, MultiPolygon]
SingleOrMultiLineString = Union[LineString, MultiLineString]


def get_polygon_from_x_y_range(x_range, y_range):
    return Polygon(
        [(x_range[0], y_range[0]), (x_range[0], y_range[1]), (x_range[1], y_range[1]), (x_range[1], y_range[0])]
    )


def get_polygon_boundary_for_data(data: pd.DataFrame, cols_to_consider=['x', 'y']):
    if cols_to_consider is None:
        cols_to_consider = data.columns
    min_max_pairs = []
    for col in cols_to_consider:
        min_max_pairs.append(
            (data[col].min(), data[col].max())
        )
    return get_polygon_from_x_y_range(*min_max_pairs)


def get_x_y_range_for_geometry(geo: Union[SingleOrMultiPolygon, MultiDiGraph]):
    if isinstance(geo, MultiDiGraph):
        _, edf = ox.graph_to_gdfs(geo)
        x1, y1, x2, y2 = edf.total_bounds
    else:
        x1, y1, x2, y2 = geo.bounds
    return (x1, x2), (y1, y2)


def remove_if_has_point_in_collection(geo):
    if isinstance(geo, GeometryCollection):
        new_geo = [x for x in geo.geoms if (not isinstance(x, Point))]
        # Assuming here that the geometries used are Line Strings
        new_geo = MultiLineString(new_geo)
        return new_geo
    return geo


def spatial_division_into_two_with_divide_dim(geo: SingleOrMultiPolygon, divide_dim=0):
    x_range, y_range = get_x_y_range_for_geometry(geo)
    x1, x2 = x_range
    y1, y2 = y_range

    if divide_dim == 1:
        xm = (x1 + x2) / 2.0
        left_poly = Polygon([(x1, y1), (x1, y2), (xm, y2), (xm, y1)])
        right_poly = Polygon([(xm, y1), (xm, y2), (x2, y2), (x2, y1)])
    else:
        ym = (y1 + y2) / 2.0
        left_poly = Polygon([(x1, y1), (x1, ym), (x2, ym), (x2, y1)])
        right_poly = Polygon([(x1, ym), (x1, y2), (x2, y2), (x2, ym)])

    left = geo.intersection(left_poly)
    right = geo.intersection(right_poly)

    left = remove_if_has_point_in_collection(left)
    right = remove_if_has_point_in_collection(right)

    return [left, right]


def spatial_division_into_n_children(geo: SingleOrMultiPolygon, n_children=2):
    assert n_children >= 2

    x_range, y_range = get_x_y_range_for_geometry(geo)
    xmin, xmax = x_range
    ymin, ymax = y_range

    h = ymax - ymin
    n_rows = int(np.sqrt(n_children))

    w = xmax - xmin
    n_cols = int(n_children / n_rows)

    offset_x = float(w) / n_cols
    offset_y = float(h) / n_rows

    childrens = []
    for x in np.linspace(0, w, n_cols + 1)[:-1]:
        for y in np.linspace(0, h, n_rows + 1)[:-1]:
            x1 = x + xmin
            x2 = x1 + offset_x
            y1 = y + ymin
            y2 = y1 + offset_y
            polygon = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
            child_geo = geo.intersection(polygon)
            child_geo = remove_if_has_point_in_collection(child_geo)
            childrens.append(child_geo)

    return childrens


def random_points_in_poly(p: Polygon, n):
    x1, y1, x2, y2 = p.bounds
    n_samples_remain = n
    ans = np.array([])
    while n_samples_remain > 0:
        x = np.random.uniform(x1, x2, size=n_samples_remain)
        y = np.random.uniform(y1, y2, size=n_samples_remain)
        points = np.array(list(map(Point, zip(x, y))))
        points = points[p.contains(points)]
        ans = np.concatenate([ans, points], axis=0)
        n_samples_remain -= len(points)
    return ans


def random_points_in_geometry(geom: Union[SingleOrMultiPolygon, SingleOrMultiLineString], n):
    if isinstance(geom, SingleOrMultiLineString):
        ans = geom.interpolate(np.random.uniform(low=0, high=1, size=n), normalized=True)
        return ans
    elif type(geom) == Polygon:
        return random_points_in_poly(geom, n)
    else:
        prob = np.array([p.area for p in geom.geoms])
        prob = prob / prob.sum()
        p_indices = np.random.choice(np.arange(len(prob)), size=n, p=prob)
        p_indices, p_counts = np.unique(p_indices, return_counts=True)

        ans = np.array([])
        for p_idx, p_c in tqdm(zip(p_indices, p_counts), desc='sampling from mp', disable=True):
            points = random_points_in_poly(geom.geoms[p_idx], p_c)
            ans = np.concatenate([ans, points], axis=0)
        np.random.shuffle(ans)
        return ans


def convert_points_to_df(points: List[Point], cols=['x', 'y']):
    arr = [[p.x, p.y] for p in points]
    return pd.DataFrame(arr, columns=cols)


def intersect_geo_series_with_poly_and_remove_empty(gs: GeoSeries, poly: SingleOrMultiPolygon):
    ans = gs.intersection(poly)
    ans = ans[~ans.is_empty]
    return ans


def rectangle_coords_to_polygon_geo_series(x1, x2, y1, y2):
    x1_y1 = np.stack([x1, y1], axis=1)
    x1_y2 = np.stack([x1, y2], axis=1)
    x2_y2 = np.stack([x2, y2], axis=1)
    x2_y1 = np.stack([x2, y1], axis=1)
    polygon_ends = np.stack([x1_y1, x1_y2, x2_y2, x2_y1], axis=1)
    return GeoSeries([Polygon(x) for x in polygon_ends])


def rectangle_queries(domain: SingleOrMultiPolygon, n: int = None, q_size="medium",
                      intersecting_with_domain_is_sufficient=False):
    q_size_percentage_dict = {
        'small': np.array([0.01, 0.1]),
        'medium': np.array([0.1, 1.0]),
        'large': np.array([1.0, 10.0])
    }
    assert q_size in q_size_percentage_dict
    q_size_percentage = q_size_percentage_dict[q_size]

    if n is None:
        q_size_n_dict = {
            'small': 10000,
            'medium': 5000,
            'large': 1000
        }
        n = q_size_n_dict[q_size]
        # print(f"Using default value n={n}, for query size={q_size}")

    x_range, y_range = get_x_y_range_for_geometry(domain)
    n_samplings_left = n

    q_x_size = np.sqrt(q_size_percentage) * (x_range[1] - x_range[0]) / 10.0
    q_y_size = np.sqrt(q_size_percentage) * (y_range[1] - y_range[0]) / 10.0

    ans = GeoSeries()
    with tqdm(total=n, desc=f"Rectangular queries for size {q_size}") as pbar:
        while (n_samplings_left):
            # print(f"Sampling queries remain = {n_samplings_left}")

            # Starting point of the xy coordinates
            query_x1 = np.random.uniform(x_range[0], x_range[1] - q_x_size[0], size=n_samplings_left)
            query_y1 = np.random.uniform(y_range[0], y_range[1] - q_y_size[0], size=n_samplings_left)

            # second x coordinate
            query_x_size_samples = np.random.uniform(q_x_size[0], q_x_size[1], size=n_samplings_left)
            query_x2 = query_x1 + query_x_size_samples

            query_y_size_samples = np.random.uniform(q_y_size[0], q_y_size[1], size=n_samplings_left)
            query_y2 = query_y1 + query_y_size_samples

            polygons = rectangle_coords_to_polygon_geo_series(query_x1, query_x2, query_y1, query_y2)

            if intersecting_with_domain_is_sufficient:
                polygons = polygons[polygons.intersects(domain)]
            else:
                polygons = intersect_geo_series_with_poly_and_remove_empty(polygons, domain)

            n_samplings_left -= len(polygons)
            ans = pd.concat([ans, polygons]).reset_index(drop=True)

            # Update tqdm
            pbar.update(len(polygons))

    return ans


def _older_count_geo_series_geometry_contains_geo_series_points(gs_geoms: GeoSeries, gs_points: GeoSeries):
    counts = np.array([geom.contains(gs_points).sum() for geom in gs_geoms.geometry])
    return pd.Series(counts, index=gs_geoms.index)


def count_geo_series_geometry_contains_geo_series_points(gs_geoms: GeoSeries, gs_points: GeoSeries):
    q_df = gpd.GeoDataFrame()
    q_df['queries'] = gs_geoms
    q_df = q_df.set_geometry('queries')
    q_df['q_id'] = gs_geoms.index

    p_df = gpd.GeoDataFrame()
    p_df['points'] = gs_points
    p_df = p_df.set_geometry('points')

    # TODO: Below might be needed for osmnx based geometries
    # p_df.set_crs(q_df.crs, inplace=True)

    temp = gpd.sjoin(q_df, p_df, how='left', predicate='contains')
    return temp.groupby('q_id')['index_right'].count()


def count_geo_series_points_contained_in_geo_series_geometry(gs_points: gpd.GeoSeries, gs_geoms: gpd.GeoSeries):
    q_df = gpd.GeoDataFrame()
    q_df['queries'] = gs_geoms
    q_df = q_df.set_geometry('queries')

    p_df = gpd.GeoDataFrame()
    p_df['points'] = gs_points
    p_df = p_df.set_geometry('points')
    p_df['p_id'] = gs_points.index.values

    temp = gpd.sjoin(p_df, q_df, how='left', predicate='within')

    temp = temp.groupby('p_id')['index_right'].count()

    temp = temp[temp > 0]

    return len(temp), temp.index.values


def get_unit_rectangle():
    return Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])


def get_gowalla_geometry():
    gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    gdf = gdf[gdf['continent'] != 'Antarctica']
    geometry = unary_union(gdf['geometry'].values)
    return geometry


def get_ny_state_geometry():
    temp = gpd.read_file(geodatasets.get_path('nybb'))
    temp = temp.geometry.to_crs('epsg:4632')
    return unary_union(temp)


def query_error_by_type(true_answers, answers, query_df: gpd.GeoDataFrame, n: int, prefix=None):
    ans = {}
    rel_err_threshold = max(1.0, 0.001 * n)  # 0.1% of dataset cardinality
    for qs in ['small', 'medium', 'large']:
        indices = query_df[query_df['size'] == qs].index
        query_error_dict = find_difference_errors_in_two_arrays(
            answers[indices], true_answers[indices],
            rel_err_threshold=rel_err_threshold
        )
        prefix_str = qs if prefix is None else f"{prefix}_{qs}"
        ans.update(dict_with_prefix(query_error_dict, prefix_str))

    return ans

# def get_cached_query_result_or_calculate_and_cache(
#         dataset_name,
#         time,
#         queries: GeoSeries,
#         points: GeoSeries,
#         base_cache_dir="D:/results/cached/query"
# ):
#     query_result_file = os.path.join(base_cache_dir, dataset_name, f"{time}.p")
#     if not os.path.exists(query_result_file):
#         print("Cached result file does not exists.")
#         print("Evaluating queries and creating ", query_result_file)
#         query_results = count_geo_series_geometry_contains_geo_series_points(queries, points)
#         query_results.to_pickle(query_result_file)
#     else:
#         query_results = pd.read_pickle(query_result_file)
#
#     return query_results
#
#
# def get_cached_priv_tree_or_calculate_and_cache(
#         dataset_name,
#         time,
#         queries: GeoSeries,
#         points: GeoSeries,
#         base_cache_dir="D:/results/cached/privtree"
# ):
#     query_result_file = os.path.join(base_cache_dir, dataset_name, f"{time}.p")
#     if not os.path.exists(query_result_file):
#         print("Cached result file does not exists.")
#         print("Evaluating queries and creating ", query_result_file)
#         query_results = count_geo_series_geometry_contains_geo_series_points(queries, points)
#         query_results.to_pickle(query_result_file)
#     else:
#         query_results = pd.read_pickle(query_result_file)
#
#     return query_results
