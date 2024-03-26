import os
import pathlib
import numpy as np
from datetime import datetime

from mad.registry import task_registry


def get_base_path():
    """Helper function to get base path.
    Required because Ray changes the working directory."""
    if 'TUNE_ORIG_WORKING_DIR' in os.environ:
        base_path = os.getenv("TUNE_ORIG_WORKING_DIR")
    else:
        base_path = ''
    return base_path


def parse_path(
    path: str,
    element_sep: str = '_',
    key_value_sep: str = '-'
) -> dict:
    """
    Parse a path into a dictionary of the key-value pairs in the path.
    E.g., t-CR_sl-128_... would be parsed into {'task': 'in-context-recall', 'seq-len': 128, ...}

    Args:
        path (str): Path to parse.
        element_sep (str, optional): Separator between key-value pairs.
        key_value_sep (str, optional): Separator between key and value.

    Returns:
        dict: Parsed path.
    """
    if os.path.isfile(path):
        path = pathlib.PurePath(path).parent.name
    else:
        path = pathlib.PurePath(path).name
    path_elements = path.split(element_sep)
    keys = [e.split(key_value_sep)[0] for e in path_elements]
    values = [f'{key_value_sep}'.join(e.split(key_value_sep)[1:]) for e in path_elements]
    parsed_path = {}
    for k, v in zip(keys, values):
        if k in SHORTHAND_TO_KEY:
            k = SHORTHAND_TO_KEY[k]
        if k == 'task':
            try:
                task_shorthands = np.array([e['shorthand'] for e in task_registry.values()])
                tasks = list(task_registry.keys())
                v = tasks[np.where(task_shorthands==v)[0][0]]
            except:
                None
        if '#' in v:
            v = float(v.replace('#', '.'))
        elif is_bool(v):
            v = bool(v)
        parsed_path[k] = v
    return parsed_path

# Mapping of key names to their shorthands used in paths:
KEY_TO_SHORTHAND = {
    'task': 't',
    'vocab_size': 'vs',
    'seq_len': 'sl',
    'num_train_examples': 'ntr',
    'num_test_examples': 'nte',
    'k_motif_size': 'km',
    'v_motif_size': 'vm',
    'frac_noise': 'fn',
    'noise_vocab_size': 'nvs',
    'multi_query': 'mq',
    'num_tokens_to_copy': 'ntc',
    'seed': 's',
    'dim': 'd',
    'layers': 'lyr',
    'lr': 'lr',
    'weight_decay': 'wd',
    'epochs': 'e',
    'batch_size': 'bs',
    'optimizer': 'opt',
    'scheduler': 'sch'
}
# Mapping shorthands back to their key names:
SHORTHAND_TO_KEY = {v: k for k,v in KEY_TO_SHORTHAND.items()}

def is_num(v) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False

def is_bool(v) -> bool:
    return v in {'True', 'False'} or isinstance(v, bool)

def make_log_path(
    base_path: str,
    mad_config,
    model_id: str,
    add_timestamp: bool=False,
    **kwargs
) -> str:
    """
    Make a log path for a given MADConfig.
    
    Args:
        base_path (str): Base log path.
        mad_config (MADConfig): MAD configuration.
        model_id (str): ID used to identify the model.
        add_timestamp (bool, optional): If yes, add timestamp to the end of the returned path.
        
    Returns:
        str: Log path.
    """
    if mad_config.task in task_registry:
        task = task_registry[mad_config.task]["shorthand"]
    else:
        task = mad_config.task
        
    path = f't-{task}_'
    for k in [
        'vocab_size',
        'seq_len',
        'num_train_examples',
        'num_test_examples',
        'k_motif_size',
        'v_motif_size',
        'multi_query',
        'frac_noise',
        'noise_vocab_size',
        'num_tokens_to_copy',
        'batch_size',
        'epochs',
        'lr',
        'weight_decay',
        'optimizer',
        'scheduler',
        'seed',
    ]:
        v = getattr(mad_config, k)
        if v is not None:
            if is_num(v):
                v = str(v).replace('.', '#')
            if is_bool(v):
                v = int(bool(v))
            path += f'{KEY_TO_SHORTHAND[k]}-{v}_'

    path += f'model-{model_id}_'

    for k,v in kwargs.items():
        path += f'{k}-{v}_'

    if add_timestamp:
        date = datetime.now().strftime("%b-%d-%Y-%H-%Mh")    
        path += f'date-{date}'
    else:
        path = path[:-1] # exclude last '_'

    return os.path.join(get_base_path(), base_path, path)


def make_dataset_path(mad_config, **kwargs):
    """Make a dataset path from MADConfig and additional kwargs."""
    if mad_config.task in task_registry:
        task = task_registry[mad_config.task]["shorthand"]
    else:
        task = mad_config.task
    path = f't-{task}_'
    for k in [
        'vocab_size',
        'seq_len',
        'num_train_examples',
        'num_test_examples',
        'k_motif_size',
        'v_motif_size',
        'multi_query',
        'frac_noise',
        'noise_vocab_size',
        'num_tokens_to_copy',
        'seed'
    ]:
        v = getattr(mad_config, k)
        if v is not None:
            if is_num(v):
                v = str(v).replace('.', '#')
            if is_bool(v):
                v = int(bool(v))
        path += f'{KEY_TO_SHORTHAND[k]}-{v}_'
    
    for k,v in kwargs.items():
        path += f'{k}-{v}_'
    
    path = path[:-1] # exclude last '_"
    return os.path.join(get_base_path(), mad_config.data_path, path)