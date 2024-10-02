import argparse
import sys
import warnings

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelSummary
from src.models.exp_jnf import JNFExp
from src.models.models import FTJNF
from src.data.datamodule import HDF5DataModule
from typing import Optional
import yaml
import os

EXP_NAME = 'JNF'

import src.utils.utils as u
u.set_printoptions_numpy()


def setup_logging(tb_log_dir: str, version_id_: Optional[int] = None):
    """
    Set-up a Tensorboard logger.

    :param tb_log_dir: path to the log dir
    :param version_id_: the version id (integer). Consecutive numbering is used if no number is given.
    """

    if version_id_ is None:
        tb_logger_ = pl_loggers.TensorBoardLogger(tb_log_dir, name=EXP_NAME, log_graph=False)

        # get current version id
        version_id_ = int(tb_logger_.log_dir.split('_')[-1])
    else:
        tb_logger_ = pl_loggers.TensorBoardLogger(tb_log_dir, name=EXP_NAME, log_graph=False, version=version_id_)

    print(f'Logging to {tb_logger_.log_dir}')

    return tb_logger_, version_id_


def load_model(ckpt_file_: str, _config: dict):
    print(f'Loading model from checkpoint {ckpt_file_}')
    model_ = JNFExp.load_from_checkpoint(ckpt_file_)
    if sys.platform == 'linux':
        model_.to('cuda')
    return model_


def get_trainer(devices, logger, max_epochs, gradient_clip_val, gradient_clip_algorithm, strategy, accelerator,
                **kwargs):
    return pl.Trainer(enable_model_summary=True,
                      logger=logger,
                      devices=devices,
                      log_every_n_steps=1,
                      max_epochs=max_epochs,
                      gradient_clip_val=gradient_clip_val,
                      gradient_clip_algorithm=gradient_clip_algorithm,
                      strategy=strategy,
                      accelerator=accelerator,
                      callbacks=[
                          # setup_checkpointing(),
                          ModelSummary(max_depth=2)
                      ],

                      )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help="Run model in test mode.")
    args = parser.parse_args()

    is_test_mode = args.test

    config_name = 'jnf_config.yaml'
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    config_file_path = os.path.join(file_dir, '..', 'config', config_name)

    with open(config_file_path) as config_file:
        config = yaml.safe_load(config_file)

    ## CONFIGURE EXPERIMENT
    ckpt_file = config['training'].get('resume_ckpt', None)
    if ckpt_file is not None:
        if not os.path.exists(ckpt_file):
            warnings.warn(f'Checkpoint file {ckpt_file} does not exist. Training from scratch.')
            ckpt_file = None
    else:
        if is_test_mode:
            raise ValueError('Testing mode, but no checkpoint file given. Check the config file.')

    ## REPRODUCIBILITY
    pl.seed_everything(config.get('seed', 0), workers=True)

    ## LOGGING
    version_id = None
    if ckpt_file is not None:
        for i, s in enumerate(ckpt_file.split(os.sep)):
            if s.startswith('version'):
                version_id = int(s.split('_')[-1])
                break
    tb_logger, version = setup_logging(config['logging']['tb_log_dir'], version_id)

    ## DATA
    data_config = config['data']
    stft_length = data_config.get('stft_length_samples', 512)
    stft_shift = data_config.get('stft_shift_samples', 256)
    dm = HDF5DataModule(**data_config)

    ## CONFIGURE EXPERIMENT
    if ckpt_file is not None:
        exp = load_model(ckpt_file, config)
    else:
        model = FTJNF(**config['network'])
        exp = JNFExp(model=model,
                     stft_length=stft_length,
                     stft_shift=stft_shift,
                     **config['experiment'])

    ## TRAIN or TEST
    trainer = get_trainer(logger=tb_logger, **config['training'])
    if not is_test_mode:
        trainer.fit(exp, dm)
    else:
        trainer.test(exp, datamodule=dm)
