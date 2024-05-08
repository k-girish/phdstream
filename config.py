import os
import geopandas as gpd
import osmnx as ox
from shapely import unary_union

from src.utils.spatial import get_gowalla_geometry, get_unit_rectangle, get_ny_state_geometry


class Config:
    DATA_PATH = "data"
    SAVE_PATH = "results"

    def __init__(self, dp=None, sp=None):
        self.DATA_PATH = dp if dp is not None else self.DATA_PATH
        self.SAVE_PATH = sp if sp is not None else self.SAVE_PATH


def gowalla_config(sensitivity=2, with_deletion=False, dp=None, sp=None):
    ans = Config(dp, sp)
    if with_deletion:
        data_prefix = "gowalla_with_deletion_30_days"
    else:
        data_prefix = "gowalla"

    arg_save_prefix = f"{data_prefix}_user_limit_{sensitivity}"

    ans.arg_save_prefix = arg_save_prefix
    ans.arg_train_file = os.path.join(ans.DATA_PATH, "gowalla", f"{arg_save_prefix}.p")
    ans.geometry = get_gowalla_geometry()
    ans.query_df = gpd.read_file(os.path.join(ans.DATA_PATH, "gowalla", "gowalla_queries.shp"))
    ans.landscape_subplots = True
    ans.n_plotting_steps = None

    return ans


def ny_state_config(dp=None, sp=None):
    ans = Config(dp, sp)
    arg_save_prefix = "ny_state"

    ans.arg_save_prefix = arg_save_prefix
    ans.arg_train_file = os.path.join(ans.DATA_PATH, arg_save_prefix, f"{arg_save_prefix}.p")
    ans.geometry = get_ny_state_geometry()
    ans.query_df = gpd.read_file(os.path.join(ans.DATA_PATH, arg_save_prefix, f"{arg_save_prefix}_queries.shp"))
    ans.landscape_subplots = False
    ans.n_plotting_steps = None
    return ans


def ny_mht_road_network_config(dp=None, sp=None):
    ans = Config(dp, sp)
    arg_save_prefix = "ny_mht"

    ans.arg_save_prefix = arg_save_prefix
    ans.arg_train_file = os.path.join(ans.DATA_PATH, arg_save_prefix, f"{arg_save_prefix}.p")
    graph = ox.graph_from_place('Manhattan, United states', network_type='drive', simplify=True)
    _, gdf = ox.graph_to_gdfs(graph)
    ans.geometry = unary_union(gdf.geometry.buffer(5e-5))
    ans.query_df = gpd.read_file(os.path.join(ans.DATA_PATH, arg_save_prefix, f"{arg_save_prefix}_queries.shp"))
    ans.landscape_subplots = False
    ans.n_plotting_steps = None
    return ans


def toy_dataset_config(dataset_name, dp=None, sp=None):
    ans = Config(dp, sp)

    dataset_name = f"motion_{dataset_name}"
    ans.arg_save_prefix = dataset_name

    ans.arg_train_file = os.path.join(ans.DATA_PATH, dataset_name, f"{dataset_name}.p")

    ans.geometry = get_unit_rectangle()

    # Both with and without deletion uses same file for queries
    if dataset_name.endswith("_with_deletion"):
        dataset_name = dataset_name[:-len("_with_deletion")]
    ans.query_df = gpd.read_file(os.path.join(ans.DATA_PATH, dataset_name, f"{dataset_name}_queries.shp"))

    ans.landscape_subplots = True
    ans.n_plotting_steps = None

    return ans


def get_config_from_data_type(dt: str, dp=None, sp=None):
    if dt == "gowalla":
        return gowalla_config(dp, sp)
    elif dt == "gowalla_with_deletion":
        return gowalla_config(dp=dp, sp=sp, with_deletion=True)
    elif dt in ["circles", "blobs", "circles_with_deletion", "blobs_with_deletion"]:
        return toy_dataset_config(dataset_name=dt, dp=dp, sp=sp)
    elif dt == "ny_state":
        return ny_state_config(dp, sp)
    elif dt == "ny_mht":
        return ny_mht_road_network_config(dp, sp)
    else:
        raise ValueError(f"Unknown dt {dt}.")
