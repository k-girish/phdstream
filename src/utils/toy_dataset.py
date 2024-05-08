import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets

from src.utils import partition_df, dict_with_prefix

DATA_TYPE_BLOBS = "blobs"
DATA_TYPE_CIRCLES = "circles"
DATA_TYPE_URBAN_RURAL = "urban_rural"


def data_gen_helper(df, n_holdout_samples):
    df, holdout_df = partition_df(df, n_holdout_samples)
    return data_info_helper(df, holdout_df)


def data_info_helper(df, holdout_df=None):
    x_range = (df['x'].min(), df['x'].max())
    y_range = (df['y'].min(), df['y'].max())

    result_dict = {
        "df": df,
        "holdout_df": holdout_df,
        "x_range": x_range,
        "y_range": y_range
    }
    return result_dict


def get_blobs(n_total, holdout_ratio, centers=3, need_labels=False):
    n_holdout_samples = int(n_total * holdout_ratio)
    data, labels = datasets.make_blobs(n_samples=n_total, centers=centers)
    df = pd.DataFrame(data, columns=['x', 'y'])
    if need_labels:
        df['label'] = labels
    result_dict = data_gen_helper(df, n_holdout_samples)
    return result_dict


def get_concentric_circles(n_total, holdout_ratio, noise_ratio=0.05, factor=0.667,
                           need_labels=False):
    n_holdout_samples = int(n_total * holdout_ratio)
    data, labels = datasets.make_circles(n_samples=n_total, noise=noise_ratio, factor=factor)
    df = pd.DataFrame(data, columns=['x', 'y'])
    if need_labels:
        df['label'] = labels
    result_dict = data_gen_helper(df, n_holdout_samples)
    return result_dict


def get_urban_and_rural(n_total, holdout_ratio, rural_urban_population_ratio=0.25, urban_rural_size_factor=3.0):
    n_holdout_samples = int(n_total * holdout_ratio)
    df = urban_and_rural(n_urban=n_total, rural_urban_population_ratio=rural_urban_population_ratio,
                         urban_rural_size_factor=urban_rural_size_factor)
    result_dict = data_gen_helper(df, n_holdout_samples)
    return result_dict


def get_sklearn_datasets(type, n_total, holdout_ratio, need_labels=False):
    if type == DATA_TYPE_BLOBS:
        return get_blobs(n_total=n_total, holdout_ratio=holdout_ratio, need_labels=need_labels)
    elif type == DATA_TYPE_CIRCLES:
        return get_concentric_circles(n_total=n_total, holdout_ratio=holdout_ratio,
                                      need_labels=need_labels)
    elif type == DATA_TYPE_URBAN_RURAL:
        return get_urban_and_rural(n_total=n_total, holdout_ratio=holdout_ratio)
    return None


def blob_and_circles():
    data, labels = datasets.make_circles(n_samples=int(1e4), noise=0.05, factor=0.667)
    df = pd.DataFrame(data, columns=['x', 'y'])
    df['x'] = 2 * df['x'] + 6.0
    df['y'] = df['y'] * 2

    data, labels = datasets.make_blobs(n_samples=int(1e4), centers=[(.0, .0)])
    df_2 = pd.DataFrame(data, columns=['x', 'y'])

    df = pd.concat([df_2, df]).reset_index(drop=True)
    return df


def urban_and_rural(n_urban=int(1e4), rural_urban_population_ratio=0.25, urban_rural_size_factor=3.0):
    data, labels = datasets.make_blobs(n_samples=n_urban, centers=[(.0, .0)])
    df = pd.DataFrame(data, columns=['x', 'y'])

    # Creating a rectangular sampling with 5 times the dimension of blob
    n_samples = int(n_urban * rural_urban_population_ratio)
    x_min = df['x'].min()
    x_max = df['x'].max()
    y_min = df['y'].min()
    y_max = df['y'].max()
    x_range = (x_max - x_min)
    y_range = (y_max - y_min)
    x_points = urban_rural_size_factor * np.random.uniform(size=n_samples) * x_range
    y_points = urban_rural_size_factor * np.random.uniform(size=n_samples) * y_range
    data = np.concatenate([x_points.reshape(-1, 1), y_points.reshape(-1, 1)], axis=1)
    df_2 = pd.DataFrame(data, columns=['x', 'y'])

    # Position the blob
    df['x'] = df['x'] - urban_rural_size_factor * x_range - x_min
    df['y'] = df['y'] - y_min

    df = pd.concat([df_2, df]).reset_index(drop=True)
    return df


def create_toy_sklearn_datasets(n, n_datasets, base_path="D:/data"):
    for data_type in [DATA_TYPE_BLOBS, DATA_TYPE_CIRCLES, DATA_TYPE_URBAN_RURAL]:
        for data_sample_iter in range(n_datasets):
            data_dict = get_sklearn_datasets(data_type, 2 * n, holdout_ratio=0.5)
            data_dict["df"].to_csv(
                os.path.join(base_path, f"{data_type}_{data_sample_iter}_train.csv"),
                index=False
            )
            data_dict["holdout_df"].to_csv(
                os.path.join(base_path, f"{data_type}_{data_sample_iter}_test.csv"),
                index=False
            )


def create_evolving_toy_dataset(n=int(1e4),
                                data_type=DATA_TYPE_CIRCLES,
                                percentage_of_first_label_entangled=50,
                                base_path="D:/data"):
    data_dict = get_sklearn_datasets(data_type, 2 * n, holdout_ratio=0.5, need_labels=True)
    data_df = data_dict["df"]

    # Use labels and entanglement to evolve data
    first_label = 0
    first_label_df = data_df[data_df['label'] == first_label]
    other_label_df = data_df[data_df['label'] != first_label]
    first_label_df_not_entangled, first_label_df_entangled = partition_df(
        first_label_df, int(len(first_label_df) * percentage_of_first_label_entangled / 100)
    )
    entangled_df = pd.concat([first_label_df_entangled, other_label_df])
    entangled_df = entangled_df.sample(frac=1.0).reset_index(drop=True)
    new_data_df = pd.concat([first_label_df_not_entangled, entangled_df]).reset_index(drop=True)

    new_data_df.to_csv(
        os.path.join(base_path, f"evolving_{data_type}_{percentage_of_first_label_entangled}_train.csv"),
        index=False
    )
    data_dict["holdout_df"].to_csv(
        os.path.join(base_path, f"evolving_{data_type}_{percentage_of_first_label_entangled}_test.csv"),
        index=False
    )
    data_dict["df"] = new_data_df
    return data_dict


def moving_from_one_cluster_to_others(n=int(1e4),
                                      data_type=DATA_TYPE_CIRCLES,
                                      base_path="D:/data"):
    data_dict = get_sklearn_datasets(data_type, 2 * n, holdout_ratio=0.5, need_labels=True)
    data_df = data_dict["df"]

    # Finding one cluster to beomce the first data
    first_label = 0
    first_label_df = data_df[data_df['label'] == first_label].reset_index(drop=True)

    # Randomize other label df
    other_label_df = data_df[data_df['label'] != first_label].sample(frac=1.0).reset_index(drop=True)

    assert len(first_label_df) <= len(other_label_df)

    motion_df_as_dict_list = []
    for i in range(len(first_label_df)):
        motion_df_as_dict_list.append({
            'x': first_label_df.loc[i, 'x'],
            'y': first_label_df.loc[i, 'y'],
            'move': -1
        })
        motion_df_as_dict_list.append({
            'x': other_label_df.loc[i, 'x'],
            'y': other_label_df.loc[i, 'y'],
            'move': 1
        })
    motion_df = pd.DataFrame(motion_df_as_dict_list)

    # Also adding
    first_label_df = first_label_df[['x', 'y']]
    first_label_df['move'] = 1

    # Concat everything
    new_data_df = pd.concat(
        [
            first_label_df,
            motion_df,
            other_label_df.iloc[len(first_label_df):, :]
        ]
    ).reset_index(drop=True)

    # For any remaining points, move should be 1
    new_data_df.loc[new_data_df['move'].isna(), 'move'] = 1

    new_data_df.to_csv(
        os.path.join(base_path, f"motion_{data_type}.csv"),
        index=False
    )
    return new_data_df


if __name__ == '__main__':
    # matplotlib.use('module://backend_interagg')
    # create_toy_sklearn_datasets(int(1e4), 2, base_path='/Users/girish/EverythingElse/MLData/processed')
    for dt in [DATA_TYPE_BLOBS, DATA_TYPE_CIRCLES]:
        df = moving_from_one_cluster_to_others(n=int(5e4), data_type=dt, base_path="D:/data/processed/2d/")
        # df['c'] = df['move'] == -1
        i_list = np.linspace(1, len(df), 10).astype(int)[1:-1]
        for i in i_list:
            temp = df.iloc[:i, :]
            temp = temp.groupby(['x', 'y']).sum()
            temp = temp[temp['move'] > 0].reset_index()
            # plt.scatter(df.loc[:i, 'x'], df.loc[:i, 'y'], c=df.loc[:i, 'c'], s=0.25)
            plt.scatter(temp['x'], temp['y'], s=0.25)
            plt.title(f"{dt}")
            plt.show()
