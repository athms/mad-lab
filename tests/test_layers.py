import os
import shutil
import torch
import yaml
from train import train
from mad.configs import MADConfig, MADModelConfig
from mad.registry import layer_registry
from tests import run_tests 


def test_layer_forward_pass():
    print('Testing forward pass for each registered layer type...')
    batch_size = 2
    dim = 128
    max_length = 128
    for layer_name, layer_reg in layer_registry.items():
        if 'rwkv' in layer_name: # [AT, Mar, 24] skipping RWKV for now due to some issues with loading their cuda kernels
            continue
        print(f'...now testing: {layer_name}')
        with open(layer_reg['cfg'], 'r') as f:
            layer_cfg = yaml.safe_load(f)
        layer_cfg['dim'] = dim
        layer_cfg['max_length'] = max_length
        layer = layer_reg['module'](**layer_cfg)
        inputs = torch.randn(batch_size, max_length, dim)
        # move to GPU if available
        if torch.cuda.is_available():
            inputs = inputs.to(torch.bfloat16 if not 'hyena' in layer_name else torch.float).cuda()
            layer = layer.to(torch.bfloat16 if not 'hyena' in layer_name else torch.float).cuda()
        outputs = layer(inputs)
        assert outputs.shape == inputs.shape
        assert torch.isnan(outputs).sum() == 0
    print('... passed!')


def test_train_layers():
    print('Testing training for each registered layer type...')
    cache_path = os.path.join('.', '.testing')
    mad_config = MADConfig(
        task='memorization',
        vocab_size=16,
        seq_len=64,
        num_train_examples=10,
        num_test_examples=10,
        batch_size=2,
        epochs=1,
    )
    for layer in layer_registry:
        print(f'...now testing: {layer}')
        model_config = MADModelConfig(layers=(layer,))
        model = model_config.build_model_from_registry()
        train(
            model=model,
            mad_config=mad_config,
            log_path=os.path.join(cache_path, layer)
        );
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    print('... passed!')


if __name__ == '__main__':
    run_tests()
