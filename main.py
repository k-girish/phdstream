import os
import traceback
import multiprocessing as mp
from typing import List

import numpy as np
import pandas as pd

from shapely import Point
from tqdm import tqdm

from config import get_config_from_data_type
from src.algo.phdstream import PHDStream
from src.utils import toy_dataset
from src.utils import get_unique_str_with_date_time_and_uuid
from src.utils.spatial import get_polygon_from_x_y_range, get_x_y_range_for_geometry


def one_experiment(
        params: PHDStream.Params,
        df: pd.DataFrame,
        qdf: pd.DataFrame = None,
        plot_steps=None
):
    result_dict = {}

    # Online Randomized Binning approach
    obj = PHDStream(params=params, df=df, query_df=qdf)
    obj_result = obj.fit_generate_iterative_for_given_df(plot_steps=plot_steps)

    # Updating results
    result_dict.update(obj_result)

    return result_dict


def run_experiment_for_given_df_and_param_list(
        params_list: List[PHDStream.Params],
        df: pd.DataFrame,
        idx: int,
        qdf: pd.DataFrame = None,
        result_save_path="results.csv",
        plot_steps=None
):
    result_dicts = []

    # Saving param list first
    param_list_dict = [x.as_dict() for x in params_list]
    pd.DataFrame(param_list_dict).to_csv(f"{result_save_path[:-4]}_params_pool_{idx}.csv", index=False)

    for params in tqdm(params_list, desc=f'Pool no: {idx}'):
        try:
            rd = one_experiment(params, df, qdf=qdf, plot_steps=plot_steps)
            result_dicts.append(rd)

            # Save to a temporary file when as results are available
            temp_result_df = pd.DataFrame(result_dicts)
            temp_result_df.to_csv(f"{result_save_path[:-4]}_temp_pool_{idx}.csv", index=False)

        except KeyboardInterrupt:
            print(f'----------------------- POOL {idx} terminated manually -----------------')
            traceback.print_exc()
        except Exception as e:
            print('--------------------------------------------------------------')
            print(f'----------------------- POOL {idx} failure -----------------')
            print('--------------------------------------------------------------')
            print(e)
            traceback.print_exc()
            print('--------------------------------------------------------------')
            print(f'----------------------- Failure params -----------------')
            print(params.as_dict())

    return result_dicts


def create_param_list_and_run_experiments(data_dict, result_save_path, all_results, base_result_dir,
                                          use_multi_processing=True, plot_steps=None):
    df = data_dict["df"]
    total_time = data_dict["total_time"]

    if 'points' not in df.columns:
        df['points'] = df[['x', 'y']].apply(lambda x: Point((x['x'], x['y'])), axis=1)

    global geometry
    global process_level_tqdm_disable
    global landscape_subplots

    if geometry is None:
        x_range = data_dict["x_range"]
        y_range = data_dict["y_range"]
        geometry = get_polygon_from_x_y_range(x_range, y_range)
    else:
        x_range, y_range = get_x_y_range_for_geometry(geometry)

    # Creating parameter list
    param_list = []

    for eps in tqdm(epsilons, desc="epsilons", disable=True):
        for sensitivity in tqdm(sensitivities, desc="sensitivities", disable=True):
            for md in tqdm(max_depths, desc="max depths", disable=True):
                for fb in tqdm(fanouts, desc="fanouts", disable=True):
                    # Algorithm specific hyperparams
                    for lnct in tqdm(leaf_node_count_thresholds, desc="leaf threshold", disable=True):
                        for method in tqdm(fit_methods, desc="fit methods", disable=True):
                            for counter_type in tqdm(counter_types, desc="counter types", disable=True):
                                for psac in tqdm(proportional_split_ancestor_counts,
                                                 desc="proportional split ancestor count",
                                                 disable=True):

                                    if infer_time_from_batch_sizes:
                                        for bs in tqdm(batch_sizes, desc="batch sizes", disable=True):
                                            for idr in tqdm(initialization_data_ratios,
                                                            desc="initialization data ratios",
                                                            disable=True):

                                                for _ in tqdm(range(n_experiments_per_hyperparam), desc="# experiment",
                                                              disable=True):
                                                    # Directory for saving plots and syn df
                                                    exp_unique_str = get_unique_str_with_date_time_and_uuid()
                                                    experiment_folder = os.path.join(base_result_dir, exp_unique_str)
                                                    if not os.path.exists(experiment_folder):
                                                        os.makedirs(experiment_folder)

                                                    params = PHDStream.Params(
                                                        epsilon=eps,
                                                        geometry=geometry,
                                                        method=method,
                                                        counter_type=counter_type,
                                                        infer_time_from_batch_sizes=infer_time_from_batch_sizes,
                                                        batch_size=bs,
                                                        initialization_data_ratio=idr,
                                                        leaf_node_count_threshold=lnct,
                                                        max_depth=md,
                                                        fanout_beta=fb,
                                                        data_dimension=2,
                                                        change_sensitivity=sensitivity,
                                                        n_synth_data_per_time=n_samples_per_experiment,
                                                        exp_id=exp_unique_str,
                                                        fig_save_folder=experiment_folder,
                                                        syn_df_save_folder=experiment_folder,
                                                        x_range=x_range,
                                                        y_range=y_range,
                                                        save_generated_synth_data=save_generated_synth_data,
                                                        disable_tqdm=process_level_tqdm_disable,
                                                        proportional_split_ancestor_count=psac,
                                                        plot_subplots_in_landscape=landscape_subplots
                                                    )
                                                    param_list.append(params)
                                    else:
                                        for init_time in tqdm(initialization_time_indices,
                                                              desc="initialization_time_indices",
                                                              disable=True):

                                            for _ in tqdm(range(n_experiments_per_hyperparam), desc="# experiment",
                                                          disable=True):
                                                # Directory for saving plots and syn df
                                                exp_unique_str = get_unique_str_with_date_time_and_uuid()
                                                experiment_folder = os.path.join(base_result_dir, exp_unique_str)
                                                if not os.path.exists(experiment_folder):
                                                    os.makedirs(experiment_folder)

                                                params = PHDStream.Params(
                                                    epsilon=eps,
                                                    geometry=geometry,
                                                    method=method,
                                                    counter_type=counter_type,
                                                    infer_time_from_batch_sizes=infer_time_from_batch_sizes,
                                                    total_time=total_time,
                                                    initialization_time=init_time,
                                                    batch_size=None,
                                                    initialization_data_ratio=0,
                                                    leaf_node_count_threshold=lnct,
                                                    max_depth=md,
                                                    fanout_beta=fb,
                                                    data_dimension=2,
                                                    change_sensitivity=sensitivity,
                                                    n_synth_data_per_time=n_samples_per_experiment,
                                                    exp_id=exp_unique_str,
                                                    fig_save_folder=experiment_folder,
                                                    syn_df_save_folder=experiment_folder,
                                                    x_range=x_range,
                                                    y_range=y_range,
                                                    save_generated_synth_data=save_generated_synth_data,
                                                    disable_tqdm=process_level_tqdm_disable,
                                                    proportional_split_ancestor_count=psac,
                                                    plot_subplots_in_landscape=landscape_subplots
                                                )
                                                param_list.append(params)

    print("Parameter list created")

    def _append_to_result_dict_(rd):
        rd['data_type'] = data_dict['data_type']
        rd['data_id'] = data_dict['data_id']

    results_per_pool = []
    if use_multi_processing:
        # No. of processes is min of available cores and size of params
        n_processes = min(len(param_list), max(mp.cpu_count() // 2 - 1, 2))
        print(f"Running experiments on {n_processes} processes")

        np.random.shuffle(param_list)

        # Create a queue to be used across processes
        temp_results_q = mp.Queue()

        with mp.Pool(n_processes) as pool:
            params_list_for_pools = np.array_split(param_list, n_processes)
            print("No. of processes used = ", len(params_list_for_pools))
            print("No. of tasks for each process:")
            print([x.shape for x in params_list_for_pools])
            pool_results = [pool.apply_async(run_experiment_for_given_df_and_param_list,
                                             (params_list_per_process, df, pool_idx, query_df, result_save_path,
                                              plot_steps)) for
                            pool_idx, params_list_per_process
                            in
                            enumerate(params_list_for_pools)]
            for res in tqdm(pool_results, desc="pool results"):
                result_dicts = res.get()
                for rd in result_dicts:
                    _append_to_result_dict_(rd)
                    temp_results_q.put(rd)

            pool.close()

        # After multiprocessing
        temp_results_q.put(None)
        results_per_pool = list(iter(lambda: temp_results_q.get(timeout=1), None))

    else:
        print("Running experiments on single process")
        for param in param_list:
            result_dict = one_experiment(param, df, query_df, plot_steps)
            _append_to_result_dict_(result_dict)
            results_per_pool.append(result_dict)
            print(result_dict)

    all_results.extend(results_per_pool)

    # Once we have the results from one pooling
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(result_save_path, index=False)


def run_given_csv_files_experiments(
        train_csv,
        arg_save_prefix, base_save_dir,
        data_type="",
        data_id=0,
        use_multi_processing=True,
        plot_steps=None
):
    # Some initializations
    run_unique_str = f"{arg_save_prefix}_id{data_id}_{get_unique_str_with_date_time_and_uuid()}"
    base_result_dir = os.path.join(base_save_dir, run_unique_str)
    if not os.path.exists(base_result_dir):
        os.makedirs(base_result_dir)

    print("Base folder for results: ", base_result_dir)
    result_save_path = os.path.join(base_result_dir, "results.csv")

    print("Reading data: ", train_csv)
    if train_csv.endswith('csv'):
        data = pd.read_csv(train_csv)
    else:
        data = pd.read_pickle(train_csv)
    print("Reading data complete = ", len(data))

    assert 'x' in data.columns
    assert 'y' in data.columns
    assert 'points' in data.columns
    if not infer_time_from_batch_sizes:
        assert 'time' in data.columns

    data_dict = toy_dataset.data_info_helper(data)
    data_dict['data_type'] = data_type
    data_dict['data_id'] = data_id
    if 'time' not in data.columns:
        data_dict['total_time'] = None
    else:
        data_dict['total_time'] = data['time'].max()
    all_results = []
    create_param_list_and_run_experiments(data_dict, result_save_path, all_results, base_result_dir,
                                          use_multi_processing, plot_steps=plot_steps)


if __name__ == '__main__':
    # ---------------------------- Hyper parameters --------------------------------------------------------
    # Run over a combination of hyperparams
    n_experiments_per_hyperparam = 1
    n_samples_per_experiment = 1

    epsilons = [2.0]
    sensitivities = [1]

    # initialization_time_indices = [100, 200, 300, 400]  # For gowalla without deletion
    # initialization_time_indices = [150, 200, 250, 300]  # For gowalla with deletion
    # initialization_time_indices = [2, 5, 10]  # For Ny Taxi
    # initialization_time_indices = [0]  # for Ny taxi except for METHOD_PRIV_TREE_INIT_AND_COUNTING
    # initialization_time_indices = [0, 2, 5, 10]  # Offline privtree baselines

    # Value of the threshold, \theta
    leaf_node_count_thresholds = [0]

    fit_methods = [
        PHDStream.Params.METHOD_PHDSTREAM,
        PHDStream.Params.METHOD_PRIV_TREE_INIT_AND_COUNTING
    ]
    counter_types = ['simple', 'block_8']

    # Counter type should be NONE with below methods
    # fit_methods = [PHDStream.Params.METHOD_PRIV_TREE_WITH_DATA_SO_FAR,
    #                PHDStream.Params.METHOD_PRIV_TREE_WITH_CURRENT_DATA]
    # counter_types = [None]

    max_depths = [20]
    fanouts = [2]
    use_multi_processing = True

    process_level_tqdm_disable = True
    proportional_split_ancestor_counts = [False]

    save_generated_synth_data = False

    # --------------------------------------------------------------------------------------------------------

    data_type = "circles_with_deletion"

    config_obj = get_config_from_data_type(
        data_type
    )

    infer_time_from_batch_sizes = not config_obj.data_has_time_attribute

    # Change these settings as needed
    if infer_time_from_batch_sizes:
        batch_sizes = [500]
        initialization_data_ratios = [0.1]

    # Other initializations
    geometry = config_obj.geometry
    landscape_subplots = config_obj.landscape_subplots
    query_df = config_obj.query_df

    run_given_csv_files_experiments(
        train_csv=config_obj.arg_train_file,
        arg_save_prefix=config_obj.arg_save_prefix,
        base_save_dir=config_obj.SAVE_PATH,
        data_type=config_obj.arg_save_prefix,
        use_multi_processing=use_multi_processing,
        plot_steps=config_obj.n_plotting_steps
    )
