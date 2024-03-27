import os
import shutil
from mad.configs import MADConfig
from mad.data import generate_data
from mad.registry import task_registry
from tests import run_tests 


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def test_data_gen():
    print('Testing dataset generation for each registered task....')
    cache_path = os.path.join('.', '.testing', 'data')
    for task, task_reg in task_registry.items():
        mad_config = MADConfig(task=task)
        for nw in [0, 2]:
            print(f'...now testing: {task} with num-workers={nw}...')
            data = generate_data(
                instance_fn=mad_config.instance_fn,
                instance_fn_kwargs=mad_config.instance_fn_kwargs,
                num_train_examples=2,
                num_test_examples=2,
                train_data_path=os.path.join(cache_path, 'train'),
                test_data_path=os.path.join(cache_path, 'test'),
                num_workers=nw
            )
            train_data = data['train']
            test_data = data['test']
            assert data is not None
            assert train_data is not None
            assert test_data is not None
            assert is_iterable(train_data), f'train_data is not iterable'
            assert is_iterable(test_data), f'test_data is not iterable'
            assert len(train_data) == 2, f'len(train_data)={len(train_data)} but expected 2'
            assert len(test_data) == 2, f'len(test_data)={len(test_data)} but expected 2'
            assert next(iter(train_data)) is not None, 'train_data is empty'
            assert next(iter(test_data)) is not None, 'test_data is empty'
            if os.path.isdir(cache_path):
                shutil.rmtree(cache_path)
    print('... passed!')


if __name__ == '__main__':
    run_tests()
