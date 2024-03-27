import os
import shutil
from train import train
from mad.configs import MADConfig, MADModelConfig
from mad.registry import task_registry
from tests import run_tests 


def test_instance_generation():
    print('Testing instance generation for each registered task....')
    test_settings = {
        'vocab_size': (16, 128),
        'seq_len': (64, 128),
        'k_motif_size': (1, 2),
        'v_motif_size': (1, 2),
        'frac_noise': (0.0, 0.5),
        'multi_query': ("True", "False"),
        'noise_vocab_size': (2,)
    }
    for task in task_registry:
        mad_config = MADConfig(task=task)
        mad_config.update_from_kwargs({k:v[0] for k,v in test_settings.items()})
        for k, vs in test_settings.items():
            for v in vs:
                for train in [True, False]:
                    print(f'...now testing: {task} with {k}={v} and is_training={train}')
                    setattr(mad_config, k, v)
                    kwargs = mad_config.instance_fn_kwargs
                    kwargs['is_training'] = train
                    inputs, targets = mad_config.instance_fn(**kwargs)
                    assert kwargs['seq_len']-1 <= inputs.size < kwargs['seq_len']+1,\
                        f'task={task} with {k}={v}: inputs.size={inputs.size} but kwargs[seq_len]={kwargs["seq_len"]}'
                    assert kwargs['seq_len']-1 <= targets.size < kwargs['seq_len']+1,\
                        f'task={task} with {k}={v}: targets.size={targets.size} but kwargs[seq_len]={kwargs["seq_len"]}'
    print('... passed!')


def test_train_tasks():
    print('Testing training for each registered task....')
    cache_path = os.path.join('.', '.testing')
    model_config = MADModelConfig(layers=['mh-attention','swiglu'])
    model = model_config.build_model_from_registry()
    for task in task_registry.keys():
        print(f'...now testing: {task} with default args.')
        mad_config = MADConfig(
            task=task,
            num_train_examples=10,
            num_test_examples=10,
            batch_size=2,
            epochs=1,
            noise_vocab_size=2,
        )
        _ = train(
            model=model,
            mad_config=mad_config,
            log_path=os.path.join(cache_path, task)
        )
    shutil.rmtree(cache_path)
    print('... passed!')


if __name__ == '__main__':
    run_tests()
