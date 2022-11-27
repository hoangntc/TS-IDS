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

PROJ_PATH = Path(os.path.join(re.sub("/TS-IDS.*$", '', os.getcwd()), 'TS-IDS'))
print(f'PROJ_PATH={PROJ_PATH}')

parser = argparse.ArgumentParser(prog='TS-IDS', description='TS-IDS')
parser.add_argument('--name', type=str, default='nf_bot_multi', help='dataset name', required=True)
parser.add_argument('--restore_model_dir', type=str, default='../model', help='restore model dir', required=False)
parser.add_argument('--restore_model_name', type=str, default='', help='restore model name', required=True)
parser.add_argument('--output_dir', type=str, default='../output', help='output dir', required=False)
parser.add_argument('--device', type=str, default='cuda:0', help='device', required=False)
args = parser.parse_args()
name = args.name
print(f'######### Experiment: {name} #########')

# This is for predicting the data including:
# 1. Initialize the main modules: data_module, model_module and trainer
# 2. Predict the output

config_path = str(PROJ_PATH /  'src' / f'config/{name}.json')
tsids = TSIDSPipeline(config_path=config_path)

# 1. Initialize data, model, trainer
data_module, model_module, trainer = tsids.initialize()

# 2. Predict
agent = tsids.predict(
    data_module,
    model_module, 
    args.restore_model_dir,
    args.restore_model_name,
    args.output_dir,
    args.device,
)
