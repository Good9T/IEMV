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
from PDTSPTrainer import PDTSPTrainer as Trainer

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

optimizer_params = {
    'optimizer': {
        'lr': 4e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [],
        'gamma': 0.3
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 20,
    'train_episodes': 10,
    'train_batch_size': 200,

    'logging': {
        'model_save_interval': 100,
        'img_save_interval': 200,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss.json'
        },
    },

    'model_load': {
        'enable': False,
        # 'path': './result/train20',  # directory path of pre-trained model and log files saved.
        # 'epoch': 2000,  # epoch version of pre-trained model to load.
    }
}

logger_params = {
    'log_file': {
        'desc': 'train20',
        'filename': 'log.txt'
    }
}

###########################################################################################

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params
    )

    trainer.run()

def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 1
    trainer_params['train_batch_size'] = 2

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(f"{g_key} = {globals()[g_key]}") for g_key in globals() if g_key.endswith('params')]

###########################################################################################

if __name__ == "__main__":
    main()