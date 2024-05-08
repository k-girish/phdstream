import os.path
import geopandas as gpd
import pandas as pd
import osmnx as ox
from matplotlib import pyplot as plt
from shapely import unary_union

from config import Config
from src.utils.spatial import rectangle_queries, SingleOrMultiPolygon, get_gowalla_geometry, get_unit_rectangle, \
    get_ny_state_geometry

base_save_dir = Config.DATA_PATH


def generate_and_save_queries_for_geometry(geometry: SingleOrMultiPolygon, queries_save_file, only_intersects=False):
    query_types = pd.Series()
    queries = gpd.GeoSeries()
    for qs in ["small", "medium", "large"]:
        temp_q_for_size = rectangle_queries(
            geometry, q_size=qs, intersecting_with_domain_is_sufficient=only_intersects
        )
        queries = pd.concat([queries, temp_q_for_size]).reset_index(drop=True)
        query_types = pd.concat(
            [
                query_types,
                pd.Series([qs] * len(temp_q_for_size))
            ]
        ).reset_index(drop=True)

    query_df = gpd.GeoDataFrame({'size': query_types, 'geometry': queries}).set_geometry('geometry')
    query_df.to_file(queries_save_file, index=False)
    return query_df


def generate_and_save_queries_for_gowalla(save_file=None):
    if save_file is None:
        save_file = os.path.join(base_save_dir, "gowalla", "gowalla_queries.shp")
    geometry = get_gowalla_geometry()
    query_df = generate_and_save_queries_for_geometry(
        geometry,
        queries_save_file=save_file
    )
    return query_df, save_file


def generate_and_save_queries_for_motion_dataset(dataset_type="motion_blobs", save_file=None):
    if save_file is None:
        save_file = os.path.join(base_save_dir, dataset_type, f"{dataset_type}_queries.shp")
    geometry = get_unit_rectangle()
    query_df = generate_and_save_queries_for_geometry(
        geometry,
        queries_save_file=save_file
    )
    return query_df, save_file


def generate_and_save_queries_for_ny_mht(save_file=None):
    if save_file is None:
        save_file = os.path.join(base_save_dir, "ny_mht", "ny_mht_queries.shp")
    graph = ox.graph_from_place('Manhattan, United states', network_type='drive', simplify=True)
    _, gdf = ox.graph_to_gdfs(graph)
    geometry = unary_union(gdf.geometry)
    query_df = generate_and_save_queries_for_geometry(
        geometry,
        queries_save_file=save_file,
        only_intersects=True
    )
    print("Queries generated")

    return query_df, save_file


def generate_and_save_queries_for_ny_state(save_file=None):
    prefix = "ny_state"
    if save_file is None:
        save_file = os.path.join(base_save_dir, prefix, f"{prefix}_queries.shp")
    geometry = get_ny_state_geometry()
    query_df = generate_and_save_queries_for_geometry(
        geometry,
        queries_save_file=save_file
    )
    return query_df, save_file


def plot_queries_from_file(file, n_q_to_plot=20):
    qdf = gpd.read_file(file)
    fig, ax = plt.subplots(1, 3)
    for st_idx, st in enumerate(['small', 'medium', 'large']):
        queries = qdf[qdf['size'] == st]['geometry']
        queries.sample(n=n_q_to_plot).plot(
            ax=ax[st_idx], facecolor="none", edgecolor='red', lw=0.7, linestyle='dashed', alpha=0.3
        )
        ax[st_idx].title.set_text(st)

    plt.show()
    fig.savefig('../temp.png')
    plt.close(fig)


def verify_save_queries():
    tq, sf = generate_and_save_queries_for_gowalla(save_file='temp')
    rq = gpd.read_file(sf)
    assert (~tq.geom_equals(rq)).sum() == 0
    assert tq.equals(rq)


if __name__ == '__main__':
    # NY State
    # _, file = generate_and_save_queries_for_ny_state()
    # plot_queries_from_file(file)

    # Manhattan
    _, file = generate_and_save_queries_for_ny_mht()
    # plot_queries_from_file(os.path.join(base_save_dir, "ny_mht", "ny_mht_queries.shp"), n_q_to_plot=1)
    plot_queries_from_file(file)
