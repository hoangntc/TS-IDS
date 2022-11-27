import os, sys, re, datetime, random, gzip, json, copy
import tqdm
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from itertools import accumulate
import argparse
from time import time
from math import ceil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

from torch_geometric.data import Data, LightningLinkData
from torch_geometric.loader import DataLoader
from utils import read_data

def build_datamodule(data_config):
    g_data, x, edge_index, edge_attr, y, input_train_edges, input_val_edges, input_test_edges, input_train_labels, input_val_labels, input_test_labels = read_data(data_config)
#     g_data = pd.read_pickle(
#         os.path.join(data_config['root'], data_config['ds_name']+'.pkl'))
#     x = torch.tensor(g_data['n_features'], dtype=torch.float)
#     edge_index = torch.tensor(g_data['edge_index'], dtype=torch.long)
#     edge_attr = torch.tensor(g_data['e_features'], dtype=torch.float)
#     y = torch.tensor(g_data['node_label'], dtype=torch.long)
#     input_train_edges = torch.tensor(g_data['edge_index'][:, np.where(g_data['tvt']=='train')[0]], dtype=torch.long)
#     input_train_labels = torch.tensor(g_data['edge_label'][np.where(g_data['tvt']=='train')[0]], dtype=torch.long)
#     input_val_edges = torch.tensor(g_data['edge_index'][:, np.where(g_data['tvt']=='val')[0]], dtype=torch.long)
#     input_val_labels = torch.tensor(g_data['edge_label'][np.where(g_data['tvt']=='val')[0]], dtype=torch.long)
#     input_test_edges = torch.tensor(g_data['edge_index'][:, np.where(g_data['tvt']=='test')[0]], dtype=torch.long)
#     input_test_labels = torch.tensor(g_data['edge_label'][np.where(g_data['tvt']=='test')[0]], dtype=torch.long)
    
    data_full = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )

    data_module = LightningLinkData(
        data=data_full,
        num_neighbors=[data_config['num_neighbors']] * 2,
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        input_train_edges=input_train_edges,
        input_train_labels=input_train_labels,
        input_val_edges=input_val_edges,
        input_val_labels=input_val_labels,
        input_test_edges=input_test_edges,
        input_test_labels=input_test_labels,
    )
    return data_module