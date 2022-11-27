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
args = parser.parse_args()
name = args.name
print(f'######### Experiment: {name} #########')

# This is for training the data including:
# 1. Initialize the main modules: data_module, model_module and trainer
# 2. Train the model

config_path = str(PROJ_PATH /  'src' / f'config/{name}.json')
config = utils.read_json(config_path)
tsids = TSIDSPipeline(config_dict=config)

# 1. Initialize data, model, trainer
data_module, model_module, trainer = tsids.initialize()

# 2. Train
tsids.train(data_module, model_module, trainer)

