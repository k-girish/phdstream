import matplotlib.pyplot as plt
import pandas as pd

import osmnx as ox
import numpy as np
from shapely.geometry import Point

from src.utils import read_all_csv_in_a_folder, train_test_split_df


def pre_process_ny_taxi_data():
    folder_path = "D:/data/ny_taxi/FOIL2013"
    df = read_all_csv_in_a_folder(
        folder_path,
        data_limit=int(1e5)
    )
    print(df.columns)
    print(df.shape)

    # Column name and filter
    df = df[[' pickup_longitude', ' pickup_latitude']]
    df.columns = ['x', 'y']
    remove_negative(df)
    x_range = (73.9, 74.02)
    y_range = (40.65, 40.9)
    q = f"(x>{x_range[0]}) and (x<{x_range[1]}) and (y>{y_range[0]}) and (y<{y_range[1]})"
    df = df.query(q)
    print(df.shape)

    train_df, test_df = train_test_split_df(df, train_ratio=0.8)
    print(train_df.shape, test_df.shape)

    train_df = train_df[df.columns]
    test_df = test_df[df.columns]

    train_df.to_csv("D:/data/ny_taxi_train.csv", index=False)
    test_df.to_csv("D:/data/ny_taxi_test.csv", index=False)

    plt.scatter(train_df.iloc[:, 0], train_df.iloc[:, 1], s=0.1)
    plt.show()


def remove_negative(df):
    if df.iloc[0, :]['x'] < 0:
        df.loc[:, 'x'] = df.loc[:, 'x'] * (-1)
    if df.iloc[0, :]['y'] < 0:
        df.loc[:, 'y'] = df.loc[:, 'y'] * (-1)


def plot_yellow_trip_data():
    df = pd.read_csv("D:/data/yellow_tripdata_2013-02_train.csv")
    x_range = (-74.3, -73.4)
    y_range = (40.5, 41)
    q = f"(x>{x_range[0]}) and (x<{x_range[1]}) and (y>{y_range[0]}) and (y<{y_range[1]})"
    df = df.query(q)
    plt.scatter(df.loc[:, 'x'].values, df.loc[:, 'y'].values, s=0.1)
    plt.show()


# ------------------------------------------ Graph Data Processing
# ------------------------------------------ Graph Data Processing
# ------------------------------------------ Graph Data Processing

def points_close_to_graph_edges(g, points, tol=1e-3):
    el, d = ox.distance.nearest_edges(g, points[:, 0], points[:, 1], return_dist=True)
    d = np.array(d)
    near_edge_points = points[d <= tol]
    far_points = points[d > tol]
    print(near_edge_points.shape, far_points.shape)
    print(f"Edge points = {len(near_edge_points)}, far points = {len(far_points)}")
    return near_edge_points, far_points


def snap_points_to_edges(g, points):
    e_list = ox.distance.nearest_edges(g, points[:, 0], points[:, 1])

    ans = []
    for e_key, point in zip(e_list, points):
        point_geom = Point(point)
        edge = g.edges.get(e_key)
        if 'geometry' not in edge:
            pass
        else:
            edge_geom = edge['geometry']
            np_on_edge = edge_geom.interpolate(edge_geom.project(point_geom))
            ans.append(np_on_edge.coords)

    ans = np.array(ans).reshape(-1, 2)
    print(f"Snapped points = {len(ans)}")

    plt.scatter(ans[:, 0], ans[:, 1], s=0.1)
    plt.title("Snapped points")
    plt.show()
    return ans


def snap_points_in_graph_range_and_save(g, data, save_file, tol=1e-3):
    points, _ = points_close_to_graph_edges(g, data, tol=tol)
    df = pd.DataFrame(snap_points_to_edges(g, points), columns=['x', 'y'])
    df.to_csv(save_file, index=False)
    return df


def generate_and_save_manhattan_data():
    df = pd.read_csv("D:/data/ny_taxi_train.csv")
    holdout_df = pd.read_csv("D:/data/ny_taxi_test.csv")
    df['x'] = (-1) * df['x']
    holdout_df['x'] = (-1) * holdout_df['x']
    points_arr = df[['x', 'y']].values
    holdout_points_arr = holdout_df[['x', 'y']].values

    g = ox.graph_from_place('Manhattan, United states', network_type='drive', simplify=True)

    snap_points_in_graph_range_and_save(g, points_arr,
                                        "D:/data/ny_manhattan_taxi_train.csv")
    snap_points_in_graph_range_and_save(g, holdout_points_arr,
                                        "D:/data/ny_manhattan_taxi_test.csv")


# ------------------------------------------ Graph Data Processing
# ------------------------------------------ Graph Data Processing
# ------------------------------------------ Graph Data Processing


if __name__ == '__main__':
    pre_process_ny_taxi_data()
