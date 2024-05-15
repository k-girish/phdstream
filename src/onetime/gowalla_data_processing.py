import matplotlib.pyplot as plt
import pandas as pd

from src.utils import read_all_csv_in_a_folder, train_test_split_df


def pre_process_beijing_taxi_data(limit_data=int(1e7)):
    folder_path = "data/beijing_2008_taxi/taxi_log_2008_by_id"
    col_names = ['taxi_id', 'date_time', 'x', 'y']
    x_range = (116.15, 116.6)
    y_range = (39.75, 40.1)
    q = f"(x>{x_range[0]}) and (x<{x_range[1]}) and (y>{y_range[0]}) and (y<{y_range[1]})"
    df = read_all_csv_in_a_folder(folder_path, filter_query=q, col_names=col_names, data_limit=limit_data)
    df = df[['x', 'y']]
    print(df.columns)
    print(df.shape)
    train_df, test_df = train_test_split_df(df)
    print(train_df.shape, test_df.shape)
    train_df.to_csv("data/beijing_train.csv", index=False)
    test_df.to_csv("data/beijing_test.csv", index=False)


def remove_negative(train_df, test_df):
    if train_df.iloc[0, :]['x'] < 0:
        train_df.loc[:, 'x'] = train_df.loc[:, 'x'] * (-1)
    if train_df.iloc[0, :]['y'] < 0:
        train_df.loc[:, 'y'] = train_df.loc[:, 'y'] * (-1)
    if test_df.iloc[0, :]['x'] < 0:
        test_df.loc[:, 'x'] = test_df.loc[:, 'x'] * (-1)
    if test_df.iloc[0, :]['y'] < 0:
        test_df.loc[:, 'y'] = test_df.loc[:, 'y'] * (-1)


def plot_data():
    train_df = pd.read_csv("data/beijing_train.csv")
    test_df = pd.read_csv("data/beijing_test.csv")
    plt.scatter(train_df.loc[:, 'x'], train_df.loc[:, 'y'], s=0.1)
    plt.show()
    plt.scatter(test_df.loc[:, 'x'], test_df.loc[:, 'y'], s=0.1)
    plt.show()


if __name__ == '__main__':
    pre_process_beijing_taxi_data(limit_data=int(1e5))
    plot_data()
