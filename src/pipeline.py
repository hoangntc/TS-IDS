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

import utils
from utils import *
from dataset import build_datamodule
from trainer import build_trainer
from model import TSIDS
from inference import InferenceAgent

class TSIDSPipeline:
    def __init__(self, config_path=None, config_dict=None):
        if config_path is not None:
            self.config = utils.read_json(config_path)
        if config_dict is not None:
            self.config = config_dict
        
        seed_everything(self.config['seed'], workers=True)
   
    def initialize(self):
        data_module = build_datamodule(self.config)
        model_module = TSIDS(self.config)
        trainer, _ = build_trainer(self.config)
        return data_module, model_module, trainer

    def train(self, data_module, model_module, trainer):
        trainer.fit(model_module, data_module)

    def test(self, data_module, model_module, trainer, checkpoint_path):
        model_test = model_module.load_from_checkpoint(checkpoint_path=checkpoint_path) 
        result = trainer.test(model_test, datamodule=data_module)
    
    def get_checkpoint_paths(self):
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        paths = sorted(checkpoint_dir.glob('*.ckpt'))
        name = self.config['name']
        checkpoint_paths = [p for p in paths if f'model={name}-' in str(p)]
        return checkpoint_paths

    def predict(self, data_module, model_module, restore_model_dir, restore_model_name, output_dir, device='cuda:0', save=True):
        args_inference = {
            'config': self.config,
            'data_module': data_module,
            'model_module': model_module,
            'restore_model_dir': restore_model_dir,
            'restore_model_name': restore_model_name,
            'output_dir': output_dir,
            'device': device,
        }
        agent = InferenceAgent(**args_inference)
        agent.infer()
        if save:
            agent.save_output()
        return agent
        
#     def run_pipeline(self):
#         # Preprocess data
#         self.preprocess_data()
        
#         # Initialize data, model, trainer
#         data_module, model_module, trainer = self.initialize()
        
#         # Train
#         self.train(data_module, model_module, trainer)
        
#         # Test with all checkpoints
#         checkpoint_paths = self.get_checkpoint_paths()
#         for checkpoint_path in checkpoint_paths:
#             self.test(data_module, model_module, trainer, checkpoint_path)
        
#         # Infer with the last checkpoints
#         restore_model_dir = str(self.config['checkpoint_dir'])
#         restore_model_name = str(checkpoint_paths[-1].name)
#         output_dir = str(PROJ_PATH / 'output')
#         self.generate_embedding(data_module, model_module, restore_model_dir, restore_model_name, output_dir)