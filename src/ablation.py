import os, sys, re, datetime, random, gzip, json, copy
from tqdm import tqdm
import pandas as pd
import numpy as np
from time import time
from math import ceil
from pathlib import Path
import itertools
import argparse
import networkx as nx
from sklearn.model_selection import ParameterGrid

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

from pipeline import TSIDSPipeline
import utils

PROJ_PATH = Path(os.path.join(re.sub("/TS-IDS.*$", '', os.getcwd()), 'TS-IDS'))
print(f'PROJ_PATH={PROJ_PATH}')

parser = argparse.ArgumentParser(prog='TS-IDS', description='TS-IDS')
parser.add_argument('--name', type=str, default='nf_bot_multi', help='dataset name', required=True)
parser.add_argument('--stage', type=str, default='all', help='stage name', required=False)
parser.add_argument('--n_folds', type=int, default=5, help='number of folds', required=False)
args = parser.parse_args()
name = args.name
stage = args.stage
n_folds = args.n_folds

assert stage in ['fit', 'predict', 'all'], 'Not valid stage name!'

print(f'######### Experiment: {name} #########')

cfname2dsname = {
    'nf_bot_multi': 'NF-BoT-IoT_cv{}_graph_multi',
    'nf_bot_binary': 'NF-BoT-IoT_cv{}_graph_binary',
    'nf_ton_multi': 'NF-ToN-IoT_cv{}_graph_multi',
    'nf_ton_binary': 'NF-ToN-IoT_cv{}_graph_binary',
}
# This is for training the data including:
# 1. Initialize the main modules: data_module, model_module and trainer
# 2. Train the model

config_path = str(PROJ_PATH /  'src' / f'config/{name}.json')
config = utils.read_json(config_path)

for fold in range(n_folds):
    config_dict = copy.deepcopy(config)
    config_dict['max_epochs'] = 200
    config_dict['ds_name'] = cfname2dsname[name].format(fold)
    config_dict['name'] = config_dict['name'] + '_cv{}_ablation'.format(fold)
    config_dict['ablation'] = True
    tsids = TSIDSPipeline(config_dict=config_dict)
    
    # 1. Initialize data, model, trainer
    data_module, model_module, trainer = tsids.initialize()

    # 2. Train
    if stage == 'fit' or stage == 'all':
        tsids.train(data_module, model_module, trainer)

    # 3. Infer
    if stage == 'predict' or stage == 'all':
        prefix = '{}_{}'.format(config_dict['gcn_type'].lower(), config_dict['name'])
        checkpoints = sorted(Path(config_dict['checkpoint_dir']).glob('model={}*.ckpt'.format(prefix)))
        print('Checkpoints:', checkpoints)
        for p in checkpoints:
            agent = tsids.predict(
                data_module,
                model_module, 
                restore_model_dir=config_dict['checkpoint_dir'],
                restore_model_name=p.name,
                save=True,
                output_dir ='../output_cv',
            )


