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
from torch.nn import Linear

import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

from torch_geometric.data import Data, LightningLinkData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GATConv, TransformerConv
from utils import calc_metrics

class InferenceAgent:
    def __init__(self,
                 config,
                 data_module,
                 model_module,
                 restore_model_dir='../model',
                 restore_model_name='.ckpt',
                 save=True,
                 output_dir='../output',
                 output_fname='',
                 device='cuda:0',
                ):
        self.restore_model_dir = restore_model_dir
        self.restore_model_name = restore_model_name
        self.output_dir = output_dir
        self.output_fname = output_fname
        self.save = save

        # initial data/model
        seed_everything(config['seed'], workers=True)
        self.data_module = data_module
        self.data_module.setup()
        self.model_module = model_module
        self.device = device
        
        # load checkpoint
        map_location = lambda storage, loc: storage.cuda()
        checkpoint_path = Path(self.restore_model_dir)/self.restore_model_name
        print(f'Load checkpoint from: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model_dict = self.model_module.state_dict()
        pretrain_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
        self.model_module.load_state_dict(pretrain_dict)
        self.model_module.eval()
        self.model_module.to(self.device)
        
    def single_dataset_inference(self, loader):
        probabilities = []
        labels = []
        for batch in loader:
            batch.cuda(self.device)
            probs = self.model_module.predict(batch)
            y = batch.edge_label
        
            probs = probs.cpu().detach().numpy().tolist()
            y = y.cpu().detach().numpy().tolist()
            
            probabilities += probs
            labels += y

        print(f'Number of samples: {len(probabilities)}')
        return probabilities, labels
    
    def infer(self):
        preds = []
        gts = []
        tvts = []
        loaders = [
                ('train', self.data_module.train_dataloader()), 
                ('val', self.data_module.val_dataloader()), 
                ('test', self.data_module.test_dataloader()),
        ]
        
        with torch.no_grad():
            for ds_name, loader in loaders:
                probabilities, labels = self.single_dataset_inference(loader)
                preds += probabilities
                gts += labels
                tvts += [ds_name]*len(labels)
        print(f'Total number of samples: {len(preds)}')
        self.output = {
            'preds': preds, 
            'gts': gts,
            'tvts': tvts,
        }
        
    def save_output(self):
        if self.output_dir != '':
            #
            df = pd.DataFrame(self.output['preds'])
            df.columns = [f'probs_{i}' for i in range(df.shape[1])]
            df['gts'] = self.output['gts']
            df['tvt'] = self.output['tvts']
            self.out_df = df
            #
            if not os.path.exists(self.output_dir): os.mkdir(self.output_dir)
            if self.output_fname == '':
                self.output_fname = os.path.basename(str(self.restore_model_name)).replace('.ckpt', '.csv')
            save_path = str(Path(self.output_dir) / self.output_fname)
            print(f'Save embeddings to: {save_path}')
            if self.save:
                df.to_csv(save_path, index=False)