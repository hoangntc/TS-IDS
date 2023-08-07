import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

def build_trainer(config, phase=None):
    name = '{}_{}'.format(config['gcn_type'].lower(), config['name'])
    logger = TensorBoardLogger(save_dir="../tb_logs")
    # callbacks
    checkpoint = ModelCheckpoint(
        dirpath=config['checkpoint_dir'], 
        filename=f'model={name}-' + '{epoch:03d}-{train_loss:.4f}-{val_loss:.4f}-{val_acc:.4f}-{val_macro_f1:.4f}-{val_weighted_f1:.4f}',
        save_top_k=config['top_k'],
        verbose=True,
        monitor=config['metric'],
        mode=config['mode'],
    )
    early_stopping = EarlyStopping(
        monitor=config['metric'], 
        min_delta=0.00, 
        patience=config['patience'],
        verbose=False,
        mode=config['mode'],
    )
    
    callbacks = [checkpoint, early_stopping]
    # trainer_kwargs
    trainer_kwargs = {
        'max_epochs': config['max_epochs'],
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': 1,
        'callbacks': callbacks,
        'log_every_n_steps': 10,
        "logger": logger,
    }

    trainer = Trainer(**trainer_kwargs)
    return trainer, trainer_kwargs