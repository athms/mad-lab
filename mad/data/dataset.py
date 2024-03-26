import os
import torch
import json
import typing as tp
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import ray
    import ray.util.multiprocessing as mp
    RAY_INSTALLED = True
except ImportError:
    print('ray not installed, falling back to python multiprocessing')
    import multiprocessing as mp
    RAY_INSTALLED = False


def check_for_leakage(train_inputs, test_inputs):
    """Helper to check for data leakage between train and test sets."""
    train_set = set([" ".join(map(str, x)) for x in train_inputs.tolist()])
    test_set = set([" ".join(map(str, x)) for x in test_inputs.tolist()])
    frac_test_in_train = 1 - (len(test_set - train_set) / len(test_set))
    if frac_test_in_train > 0.001:
        print(
            "WARNING: Potential data leakage detected. " 
            f"{frac_test_in_train: 0.2f} of test examples are in the train set."
        )


# infrastructure for datasets that are kept in memory:

def generate_data(
    instance_fn,
    instance_fn_kwargs,
    train_data_path: str,
    test_data_path: str,
    num_train_examples: int,
    num_test_examples: int,
    num_workers: int,
    verbose: bool = False
):
    """
    Generate train/test data in memory (for small datasets or large memory).

    Args:
        instance_fn (callable): function that generates individual task samples; must return tuple of (inputs, targets)
        instance_fn_kwargs (dict): keyword arguments for instance_fn
        num_train_examples (int): number of training examples to generate
        num_test_examples (int): number of test examples to generate
        train_data_path (str): path to write training data to
        test_data_path (str): path to write test data to
        num_workers (int): number of workers to use. If 0, no parallelization is used.
        verbose (bool): whether to print progress.

    Returns:
        dict: dictionary with keys 'train' and 'test' containing torch datasets
    """

    # Make Training Data.

    train_data_exists = os.path.exists(train_data_path)
    train_instance_fn_kwargs = dict(instance_fn_kwargs)
    train_instance_fn_kwargs['is_training'] = True
    training_dataset = MemoryDataset(
        instance_fn=instance_fn,
        instance_fn_kwargs=train_instance_fn_kwargs,
        verbose=verbose
    )

    if train_data_path is not None and train_data_exists:
        if verbose:
            print(f'Training data exists, loading from: {train_data_path}')
        training_dataset.load_data(train_data_path)
    
    elif train_data_path is not None and not train_data_exists:
        training_dataset.generate_data(
            num_examples=num_train_examples,
            num_workers=num_workers
        )
        training_dataset.save_data(train_data_path)
    
    else:
        training_dataset.generate_data(
            num_examples=num_train_examples,
            num_workers=num_workers
        )     

    # Make Test Data.   

    test_data_exists = os.path.exists(test_data_path)
    test_instance_fn_kwargs = dict(instance_fn_kwargs)
    test_instance_fn_kwargs['is_training'] = False
    test_dataset = MemoryDataset(
        instance_fn=instance_fn,
        instance_fn_kwargs=test_instance_fn_kwargs,
        verbose=verbose
    )

    if test_data_path is not None and test_data_exists:
        if verbose:
            print(f'Test data exists, loading from: {test_data_path}')
        test_dataset.load_data(test_data_path)
        
    elif test_data_path is not None and not test_data_exists:
        test_dataset.generate_data(
            num_examples=num_test_examples,
            num_workers=num_workers
        )
        test_dataset.save_data(test_data_path)

    else:    
        test_dataset.generate_data(
            num_examples=num_test_examples,
            num_workers=num_workers
        )

    # Check for data leakage.

    check_for_leakage(training_dataset.inputs, test_dataset.inputs)
    assert training_dataset.inputs.shape[0] == num_train_examples,\
        f'Training data have {training_dataset.inputs.shape[0]} samples but should have {num_train_examples} samples.'
    assert test_dataset.inputs.shape[0] == num_test_examples,\
        f'Test data have {test_dataset.inputs.shape[0]} samples but should have {num_test_examples} samples.'

    return {"train": training_dataset, "test": test_dataset}


class MemoryDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset that stores data in memory.

    Args:
        instance_fn (callable): function used to generate individual task samples; must return tuple of (inputs, targets)
        instance_fn_kwargs (dict): keyword arguments for instance_fn
        is_training (bool): whether training or test targets are used.
        verbose (bool): whether to print progress.
    """
    def __init__(self,
        instance_fn: callable,
        instance_fn_kwargs: dict,
        verbose: bool = True
    ) -> None:
        super().__init__()
        self.instance_fn = instance_fn
        self.instance_fn_kwargs = instance_fn_kwargs
        self.inputs = None
        self.targets = None
        self.verbose = verbose

    def __getitem__(self, idx):
        assert self.inputs is not None, 'no data generated yet; call self.generate_data!'
        return self.inputs[idx], self.targets[idx]

    def __len__(self):
        assert self.inputs is not None, 'no data generated yet; call self.generate_data!'
        return len(self.inputs)

    def load_data(self, path: str):
        """
        Load data from path.
        
        Args:
            path (str): path to load data from.
        """
        if self.verbose:
            print(f'Loading data from: {path}')
        assert os.path.isdir(path)
        self.inputs = np.load(os.path.join(path, 'inputs.npy'))
        self.targets = np.load(os.path.join(path, 'targets.npy'))
        assert len(self.inputs) == len(self.targets)

    def save_data(self, path: str):
        """
        Save data to path.
        
        Args:
            path (str): path to save data to.
        """
        assert self.inputs is not None
        assert self.targets is not None
        assert len(self.inputs) == len(self.targets)
        if os.path.isdir(path):
            print(f'WARNING: directory "{path}" exists already, overwriting...')
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, 'inputs.npy'), self.inputs)
        np.save(os.path.join(path, 'targets.npy'), self.targets)

    def generate_data(self,
        num_examples: int,
        num_workers: int=0,
        verbose: bool=True,
    ):
        """
        Generate data with self.instance_fn.

        Args:
            num_examples (int): number of examples to generate.
            num_workers (int): number of workers to use. If 0, no parallelization is used.
            verbose (bool): whether to print progress.
        """
        assert self.inputs is None
        assert self.targets is None

        if self.verbose:
            print(f'Generating dataset with {num_examples} examples, using {num_workers} workers...')
        
        # parallel data generation:
        if num_workers > 1:

            # with ray:
            if RAY_INSTALLED:
                kv_map_id = ray.put(self.instance_fn_kwargs.get('kv_map', None))

                @ray.remote
                def f(*args):
                    return self.instance_fn(
                        **{k:v for k,v in self.instance_fn_kwargs.items() if k!='kv_map'},
                        kv_map=ray.get(kv_map_id)
                    )
                            
            # without ray:
            else:

                def f(*args):
                    return self.instance_fn(**self.instance_fn_kwargs)
                                
            pool = mp.Pool(num_workers)
            if RAY_INSTALLED:
                # we process in num_workers chunks to avoid cpu throttle!
                # see: https://discuss.ray.io/t/using-a-subset-of-available-cpus
                instances = []
                for _ in range(0, num_examples, num_workers):
                    chunk_instances = pool.map(f.remote, range(num_workers))
                    instances.extend(ray.get(chunk_instances))
            else:
                instances = pool.map(f, range(num_examples))
            pool.close()

        # sequential data generation:
        else:
            iterator = tqdm(range(num_examples)) if verbose else range(num_examples)
            instances = [self.instance_fn(**self.instance_fn_kwargs) for _ in iterator]

        if len(instances[-1])==2:
            self.inputs, self.targets = [np.stack(i) for i in zip(*instances)]
        else:
            raise ValueError(f'returned instances need to contain 2 arrays (inputs, targets) but got {len(instances[-1])}')
    

# infrastructure for datasets that are stored on disk and read from there during training:

def generate_data_disk(
    instance_fn: callable,
    instance_fn_kwargs: dict,
    num_train_examples: int,
    num_test_examples: int,
    train_data_path: str,
    test_data_path: str,
    num_workers: int = 0,
    verbose: bool = True,
    num_docs_training: int = 1,
    num_docs_test: int = 1,
    return_datasets: bool = True,
    *args, **kwargs
):
    """
    Write data to disk and return train/test torch datasets that read data from disk
    during training.

    Args:
        instance_fn (callable): function that generates individual task samples; must return tuple of (inputs, targets)
        instance_fn_kwargs (dict): keyword arguments for instance_fn
        num_train_examples (int): number of training examples
        num_test_examples (int): number of test examples
        train_data_path (str): path to write training data to
        test_data_path (str): path to write test data to
        num_workers (int): number of workers to use. If 0, no parallelization is used.
        verbose (bool): whether to print progress.
        num_docs_training (int): number of documents to write training data to.
        num_docs_test (int): number of documents to write test data to.

    Returns:
        dict: dictionary with keys 'train' and 'test' containing torch datasets
    """
    assert not os.path.isdir(train_data_path), f'directory "{train_data_path}" already exists'
    assert not os.path.isdir(test_data_path), f'directory "{test_data_path}" already exists'

    # Make Training Data.

    train_instance_fn_kwargs = dict(instance_fn_kwargs)
    train_instance_fn_kwargs['is_training'] = True
    training_dataset = DiskDataset(
        instance_fn=instance_fn,
        instance_fn_kwargs=train_instance_fn_kwargs,
        verbose=verbose
    )
    training_dataset.generate_data(
        path=train_data_path,
        num_examples=num_train_examples,
        num_documents=num_docs_training,
        num_workers=num_workers
    )

    # Make Test Data.
    
    test_instance_fn_kwargs = dict(instance_fn_kwargs)
    test_instance_fn_kwargs['is_training'] = False
    test_dataset = DiskDataset(
        instance_fn=instance_fn,
        instance_fn_kwargs=test_instance_fn_kwargs,
        verbose=verbose
    )
    test_dataset.generate_data(
        path=test_data_path,
        num_examples=num_test_examples,
        num_documents=num_docs_test,
        num_workers=num_workers
    )

    if return_datasets:
        return {"train": training_dataset, "test": test_dataset}


class DiskDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset that stores data on disk and reads from there during training.

    Args:
        instance_fn (callable): function used to generate individual task samples; must return tuple of (inputs, targets)
        instance_fn_kwargs (dict): keyword arguments for instance_fn
        is_training (bool): whether training or test targets are used.
        verbose (bool): whether to print progress.
    """
    def __init__(self,
        instance_fn: callable,
        instance_fn_kwargs: dict,
        verbose: bool=True
    ) -> None:
        super().__init__()
        self.instance_fn = instance_fn
        self.instance_fn_kwargs = instance_fn_kwargs
        self.inputs = None
        self.targets = None
        self.documents = None
        self.verbose = verbose

    def __getitem__(self, idx: int):
        # TODO: also read data-idx from disk?
        doc_idx, doc_path = self.data_idx.iloc[idx][['doc_idx', 'doc_path']]
        instance = self.read_instance_from_doc(doc_idx, doc_path)
        return (
            np.array(instance['inputs'].split()).astype(int),
            np.array(instance['targets'].split()).astype(int),
        )

    def __len__(self):
        return len(self.data_idx)

    def use_data_from_idx(self, data_idx: tp.Union[str, pd.DataFrame]):
        """
        Use existing dataset.
        
        Args:
            data_idx (str or DataFrame): data index for existing dataset (as generated by self.make_idx).
        """
        self.data_idx = pd.read_csv(data_idx, index_col=0) if isinstance(data_idx, str) else data_idx
        assert 'doc_path' in self.data_idx.columns
        assert 'doc_idx' in self.data_idx.columns
        self.documents = set(list(self.data_idx['doc_path'].values))
        assert len(self.data_idx) == sum([len(pd.read_csv(d, index_col=0)) for d in self.documents])

    def generate_data(self,
        path: str,
        num_examples: int,
        num_documents: int=1,
        num_workers: int=0
    ):
        """
        Generate data and write to specified number of documents in path.
        
        Args: 
            path (str): path to write data to.
            num_examples (int): number of examples to generate.
            num_documents (int): number of documents to write data to.
            num_workers (int): number of workers to use for data generate. If 0, no parallelization is used.
        """
        assert not os.path.exists(path)
        assert self.documents is None
        if self.verbose:
            print(
                f'\nwriting dataset of {num_examples} samples to '
                f'{num_documents} documents in "{path}"...'
            )
        os.makedirs(path, exist_ok=True)
        self.documents = []
        num_examples_per_doc = num_examples // num_documents
        for i in range(num_documents):
            if self.verbose:
                print(f'\tcreating document {i+1}/{num_documents}...')
            doc_path = os.path.join(path, f'train_{i+1}.txt')
            assert not os.path.exists(doc_path), f'file {doc_path} already exists'
            self.generate_doc_data(
                doc_path=doc_path,
                num_examples=num_examples_per_doc if i < num_documents-1 else num_examples - num_examples_per_doc*i,
                num_workers=num_workers,
            )
            self.documents.append(doc_path)

        # index generated data
        self.make_idx(path_idx=os.path.join(path, 'data_idx.csv'))

    def generate_doc_data(self,
        doc_path: str,
        num_examples: int,
        num_workers: int=0,
    ):
        """
        Generate data and write to document.

        Args:
            doc_path (str): document path to write data to
            num_examples (int): number of examples
            num_workers (int): number of workers to use. If 0, no parallelization is used.
        """
        # generate data in parallel:
        if num_workers > 1:

            # with ray:
            if RAY_INSTALLED:
                kv_map_id = ray.put(self.instance_fn_kwargs.get('kv_map', None))

                @ray.remote
                def f(*args):
                    instance = self.instance_fn(
                        **{k:v for k,v in self.instance_fn_kwargs.items() if k!='kv_map'},
                        kv_map=ray.get(kv_map_id)
                    )
                    self.write_instance_to_doc(instance, doc_path)
                
            # without ray:
            else:
                print('ray not installed, falling back to multiprocessing')
                import multiprocessing as mp

                def f(*args):
                    instance = self.instance_fn(**self.instance_fn_kwargs)
                    self.write_instance_to_doc(instance, doc_path)
                
            pool = mp.Pool(num_workers)
            if RAY_INSTALLED:
                ray.get(pool.map(f.remote, range(num_examples)))
            else:
                pool.map(f, range(num_examples))
            pool.close()

        # generate data in sequence:
        else:
            iterator = tqdm(range(num_examples)) if self.verbose else range(num_examples)
            for _ in iterator:
                instance = self.instance_fn(**self.instance_fn_kwargs)
                self.write_instance_to_doc(instance, doc_path)

    def write_instance_to_doc(self, instance: tp.Tuple, doc_path: str):
        """
        Write instance to document.
        
        Args:
            instance (tuple): generated instance.
            doc_path (str): path of document to which instance is appended.
        """

        if len(instance)==2:
            instance_dict = {
                'inputs': " ".join([str(t) for t in instance[0]]),
                'targets': " ".join([str(t) for t in instance[1]]),
            }
        else:
            raise ValueError(f'returned instances need to contain 2 (or 3) entries but got {len(instance)}')

        with open(doc_path, 'a+') as f:
            json.dump(instance_dict, f)
            f.write('\n') # add newline for readability
                
    def make_idx(self, path_idx: str):
        """
        Index generated data and write index to path.
        
        Args: 
            path_idx (str): path to write index to.
        """
        assert not os.path.isfile(path_idx)
        i, data_idx = 0, []
        for document in self.documents:
            doc_lines = self.read_lines_from_doc(document)
            n_instances = len(doc_lines)
            data_idx.append(
                pd.DataFrame({
                    'doc_idx': np.arange(n_instances),
                    'doc_path': [document]*n_instances
                }, index=np.arange(i, i+n_instances))
            )
            i += n_instances
        data_idx = pd.concat(data_idx)
        data_idx.to_csv(path_idx)
        self.data_idx = data_idx
        
    def read_lines_from_doc(self, doc_path: str):
        with open(doc_path, 'r') as f:
            data = f.readlines()
        return [json.loads(line) for line in data]

    def read_instance_from_doc(self, idx: int, doc_path: str):
        with open(doc_path, 'r') as file:
            file.seek(0)  # move file pointer to the beginning of file
            for _ in range(idx - 1):
                file.readline()
            line_content = file.readline()
        return json.loads(line_content) if line_content else None