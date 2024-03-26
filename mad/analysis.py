import os
import yaml
import pandas as pd

from mad.configs import make_benchmark_mad_configs
from mad.registry import task_registry
from mad.paths import make_log_path, parse_path


TASKS = list(task_registry.keys())

def load_yml(path):
    """Helper function to load a yaml file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_result_paths(
    model_id: str,
    logs_path: str,
    task: str = 'all',
    check_present: bool = True
) -> list:
    """
    Get paths for all results.csv files resulting from the MAD benchmark
    for a given model and task.

    Args:
        model_id (str): The ID used for the model during training.
        logs_path (str): Path to the logs directory.
        task (str, optional): Task to get results for. 
            'all' indicates that results for all MAD tasks are retrieved.
        check_present (bool, optional): Check if the files are present; throw an error if not.

    Returns:
        list: List of paths to results.csv files. 
    """

    assert task in {'all', *task_registry.keys()}, f'{task} is not a valid task.'
    tasks = TASKS if task == 'all' else [task]

    all_mad_configs = make_benchmark_mad_configs()
    mad_configs = []
    for mc in all_mad_configs:
        if mc.task in tasks:
            mad_configs.append(mc)

    result_paths = []
    for mc in mad_configs:
        log_path = make_log_path(
            base_path=logs_path,
            mad_config=mc,
            model_id=model_id,
        )
        result_paths.append(os.path.join(log_path, 'results.csv'))
        if check_present:
            assert os.path.exists(result_paths[-1]), f'{result_paths[-1]} is missing!'

    return result_paths


def aggregate_model_results(
    model_id: str,
    logs_path: str,
    task: str = 'all',
    subset_to_best_sweep_runs: bool = True,
    metric: str = 'test_acc',
    metric_criterion: str = 'max'
) -> pd.DataFrame:
    """
    Aggregate results for a given model and task of the MAD
    benchmark.
    
    Args:
        model_id (str): The ID used for the model during training.
        logs_path (str): Path to the logs directory.
        task (str, optional): Task to get results for.
            'all' indicates that results for all MAD tasks are retrieved.
        subset_to_best_sweep_runs (bool, optional): Subset to best sweep runs.
        metric (str, optional): Metric to use for determining best sweep runs.
        metric_criterion (str, optional): Criterion for determining best sweep runs based on metric
        
    Returns:
        pd.DataFrame: Aggregated results.
    """

    result_paths = get_result_paths(
        logs_path=logs_path,
        model_id=model_id,
        task=task,
        check_present=True
    )

    results = []
    for result_path in result_paths:
        result_df = pd.read_csv(result_path)
        parsed_path = parse_path(result_path)
        parsed_path_df = pd.DataFrame(parsed_path, index=[0])
        results.append(pd.concat([parsed_path_df, result_df], axis=1))
    
    results_df = pd.concat(results).reset_index()

    # drop NaNs:
    size_before = results_df.shape[0]
    results_df = results_df.dropna()
    size_after = results_df.shape[0]
    if size_before != size_after:
        print(f'/!\ dropped {size_before - size_after} rows from results because they contained NaNs')

    if subset_to_best_sweep_runs:
        results_df = subset_model_results_to_best_sweep_runs(
            results=results_df,
            metric=metric,
            metric_criterion=metric_criterion
        )

    return results_df


def subset_model_results_to_best_sweep_runs(
    results: pd.DataFrame,
    metric: str = 'test_acc',
    metric_criterion: str = 'max'
) -> pd.DataFrame:
    """
    Subset aggregated results to best sweep runs.

    Args:
        results (pd.DataFrame): Results dataframe, as resulting from calling 'aggregate_model_results'.
        metric (str, optional): Metric to use for determining best sweep runs.
        metric_criterion (str, optional): Criterion for determining best sweep runs based on metric (one of 'min', 'max').
    
    Returns:
        pd.DataFrame: Subsetted results.
    """
    grouping_columns = results.columns
    metric_columns = [
        c for c in grouping_columns
        if c.startswith('train_')
        or c.startswith('test_')
        and c!=metric
    ]
    grouping_columns = [
        c for c in grouping_columns
        if c not in {'lr', 'weight_decay'}
        and not c in metric_columns
        and c != 'index'
    ]
    results_grouped = results.groupby(by=[c for c in grouping_columns if c!=metric])

    def find_best_performance(group):
        if metric_criterion == 'max':
            best_row = group.loc[group[metric].idxmax()]
        elif metric_criterion == 'min':
            best_row = group.loc[group[metric].idxmin()]
        else:
            raise ValueError(f'invalid metric_criterion "{metric_criterion}", must be one of "min", "max"')
        return best_row

    best_results_from_sweep = results_grouped.apply(find_best_performance)
    best_results_from_sweep = best_results_from_sweep.reset_index(drop=True).drop('index', axis=1)
    return best_results_from_sweep


def compute_model_mad_scores(
    model_id: str,
    logs_path: str,
    task: str = 'all',
    metric: str = 'test_acc',
    metric_criterion: str = 'max'
):
    """
    Compute MAD scores for a given model and task of the MAD benchmark.

    Args:
        model_id (str): Model ID.
        logs_path (str): Path to the logs directory.
        task (str, optional): Task to get results for.
            `all` indicates that MAD scores are computed for all tasks.
        metric (str, optional): Metric to use for determining best sweep runs.
        metric_criterion (str, optional): Criterion for determining best sweep runs based on metric (one of 'min', 'max').
    
    Returns:
        pd.DataFrame: MAD scores.
    """
    model_results = aggregate_model_results(
        model_id=model_id,
        logs_path=logs_path,
        task=task,
        subset_to_best_sweep_runs=True,
        metric=metric,
        metric_criterion=metric_criterion
    )
    model_results_grouped = model_results.groupby(by=['task'])

    def compute_mean(group):
        mean = group[metric].mean()
        return mean

    mad_scores = model_results_grouped.apply(compute_mean)
    return mad_scores