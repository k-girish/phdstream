import datetime
import uuid
import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.col_name_and_query import create_query_for_given_cols_and_values
import src.utils.time as time_util

from IPython.core.display import display


def partition_df(df, len_other_df, random_partition=True):
    if random_partition:
        other_df_indices = np.random.choice(len(df), size=len_other_df, replace=False)
    else:
        other_df_indices = np.arange(len(df))[:len_other_df]
    other_df = df.iloc[other_df_indices, :].copy()
    df = df.drop(other_df.index).copy()
    df = df.reset_index(drop=True)
    other_df = other_df.reset_index(drop=True)
    return df, other_df


def find_difference_errors_in_two_arrays(x, y, print_results=False, rel_err_threshold=0.001):
    float_n = float(len(x))
    abs_diff = np.abs(y - x)
    norm_2_avg_err = np.linalg.norm(abs_diff) / np.sqrt(float_n)
    max_err = np.max(abs_diff)
    avg_err = np.mean(abs_diff)

    rel_err = np.mean(np.divide(abs_diff, (x + rel_err_threshold)))

    if print_results:
        print(f"Avg = {avg_err}, Max = {max_err}, MeanRel = {rel_err}, 2-norm = {norm_2_avg_err}")

    return {
        'avg_err': avg_err,
        'max_err': max_err,
        'rel_err': rel_err,
        'norm_2_err': norm_2_avg_err
    }


def dict_with_prefix(obj, prefix: str):
    ans = {}
    for (k, v) in obj.items():
        ans[f"{prefix}_{k}"] = v
    return ans


def get_unique_str_with_date_time_and_uuid():
    return f"{datetime.datetime.now().strftime('%m_%d_%H_%M')}_{uuid.uuid1().__str__()}"


def find_save_path(sub_folder: str, prefix: str, extension: str = "csv"):
    file_name = f"{prefix}_{get_unique_str_with_date_time_and_uuid()}.{extension}"
    path = os.path.join("../../results", sub_folder, file_name)
    return path


def save_results(results: pd.DataFrame, sub_folder: str, prefix: str, path: str = None):
    if path is None:
        path = find_save_path(sub_folder, prefix)
    # print("Saving to ", path)
    results.to_csv(path)
    return path


def view_all_df(temp):
    """
    This utility function can be used to view all columns and rows of a pandas dataframe
    :param temp: the pandas dataframe to view
    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1):
        display(temp)


def read_concatenate_multiple_csv(base_path, file_name_list, cols_to_keep=None, col_names=None, filter_query=None,
                                  data_limit=None):
    df = pd.read_csv(os.path.join(base_path, file_name_list[0]), usecols=cols_to_keep, names=col_names)
    if filter_query is not None:
        df = df.query(filter_query)

    for i in tqdm(range(1, len(file_name_list))):
        # Read one file in temp and preprocess
        temp = pd.read_csv(os.path.join(base_path, file_name_list[i]), usecols=cols_to_keep, names=col_names)
        if filter_query is not None:
            temp = temp.query(filter_query)

        # Concat
        df = pd.concat([df, temp])

        # Limiting the data
        if (data_limit is not None) and (len(df) >= data_limit):
            warnings.warn("Data limit crossed, will stop reading more data!")
            df = df.iloc[:data_limit, :]
            break

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


def read_all_csv_in_a_folder(folder_path, cols_to_keep=None, col_names=None, filter_query=None, data_limit=None):
    return read_concatenate_multiple_csv(
        folder_path,
        os.listdir(folder_path),
        cols_to_keep=cols_to_keep,
        col_names=col_names,
        filter_query=filter_query,
        data_limit=data_limit)


def train_test_split_df(df, train_ratio=0.8):
    tk = time_util.start_time_record()
    train_indices = np.random.choice(len(df), size=round(train_ratio * len(df)), replace=False)
    test_indices = np.setdiff1d(np.arange(len(df)), train_indices)
    train_df = df.iloc[train_indices, :]
    test_df = df.iloc[test_indices, :]
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    time_util.end_time_record(tk, print_duration=True)
    return train_df, test_df
