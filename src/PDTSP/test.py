DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import logging

from utils import create_logger
from PDTSPTester import PDTSPTester as Tester

###########################################################################################

env_params = {
    'customer_size': 10,
    'node_size': 21,
    'problem_gen_params': {
        'scaler': 1.0,
    },
    'pomo_size': 10,
}

model_params = {
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256**0.5,
    'encoder_layer_num': 5,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16**0.5,
    'head_num': 16,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1/2)**0.5,
    'ms_layer2_init': (1/16)**0.5,
    'eval_type': 'argmax',
    'one_hot_seed_cnt': 150,
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/train20',
        'epoch': 20,
    },
    'test_episodes': 20,
    'test_batch_size': 100,
    'augmentation_enable': True,
    'aug_factor': 50,
    'aug_batch_size': 50,
}

if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test20',
        'filename': 'log.txt'
    }
}

###########################################################################################

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(
        env_params=env_params,
        model_params=model_params,
        tester_params=tester_params
    )
    tester.run()

def _set_debug_mode():
    pass

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(f"{g_key} = {globals()[g_key]}") for g_key in globals() if g_key.endswith('params')]

###########################################################################################

if __name__ == "__main__":
    main()