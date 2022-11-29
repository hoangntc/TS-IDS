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
import utils

class GNN(torch.nn.Module):
    def __init__(self, gcn_type, in_channels, hidden_channels, heads, edge_dim, dropout):
        super().__init__()
        if gcn_type == 'GAT':
            self.conv1 = GATv2Conv(
                in_channels=in_channels, 
                out_channels=hidden_channels,
                heads=heads,
                edge_dim=edge_dim,
                dropout=dropout,
            )
            self.conv2 = GATv2Conv(
                in_channels=hidden_channels*heads, 
                out_channels=hidden_channels,
                heads=heads,
                edge_dim=edge_dim,
                dropout=dropout,
            )
        elif gcn_type == 'Transformer':
            self.conv1 = TransformerConv(
                in_channels=in_channels, 
                out_channels=hidden_channels,
                heads=heads,
                edge_dim=edge_dim,
                dropout=dropout,
            )
            self.conv2 = TransformerConv(
                in_channels=hidden_channels*heads, 
                out_channels=hidden_channels,
                heads=heads,
                edge_dim=edge_dim,
                dropout=dropout,
            )

    def forward(self, x, edge_index, edge_attr):
        h1 = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        h2 = self.conv2(x=h1, edge_index=edge_index, edge_attr=edge_attr)
        return h2

class EdgeEncoder(torch.nn.Module):
    def __init__(self, operator='HAD'):
        super().__init__()
        assert operator in ['HAD', 'Concat', 'L1', 'L2']
        self.operator = operator

    def forward(self, h, edge_label_index):
        src_h = h[edge_label_index[0]]
        dst_h = h[edge_label_index[1]]
        if self.operator == 'HAD':
            link_f = src_h * dst_h
        elif self.operator == 'Concat':
            link_f = torch.cat([src_h, dst_h], dim=1)
        elif self.operator == 'L1':
            link_f = torch.abs(src_h - dst_h)
        elif self.operator == 'L2':
            link_f = (src_h - dst_h)**2
        return link_f
    
class TSIDS(pl.LightningModule):
    '''
    GNN-based intrusion detection system
    '''
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        assert self.hparams.gcn_type in ['GAT', 'Transformer'], 'Not implemented method!'
        self.class_weight = torch.tensor([float(i) for i in self.hparams.class_weight.split('-')], dtype=torch.float32)
        # GNN
        ## node encoder
        self.gnn = GNN(
            self.hparams.gcn_type,
            self.hparams.in_channels, 
            self.hparams.hidden_channels, 
            self.hparams.heads, 
            self.hparams.edge_dim, 
            self.hparams.dropout,
        )
        
        ## edge_encoder
        self.edge_encoder = EdgeEncoder(operator=self.hparams.operator)
        n_hidden_dim = self.hparams.hidden_channels * self.hparams.heads
        ## Readout
        if self.hparams.operator == 'Concat':
            # e_hidden_dim = self.hparams.edge_dim + n_hidden_dim * 2 
            e_hidden_dim = n_hidden_dim * 2 
        else:
            # e_hidden_dim = self.hparams.edge_dim + n_hidden_dim
            e_hidden_dim = n_hidden_dim
        self.lin0 = Linear(e_hidden_dim, self.hparams.out_dim * 2)
        self.lin1 = Linear(self.hparams.out_dim * 2, self.hparams.out_dim)
        self.classifier = nn.Linear(self.hparams.out_dim, self.hparams.num_labels)
        
        ## Loss
        self.loss = nn.CrossEntropyLoss(weight=self.class_weight)
        
        # SSL
        self.discriminator = Linear(n_hidden_dim, 1)
        self.loss_ssl = nn.BCELoss()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer  
    
    def forward(self, data):
        e_feature = data.edge_attr
        h2 = F.relu(self.gnn(data.x, data.edge_index, data.edge_attr))
        h3 = self.edge_encoder(h2, data.edge_label_index)
        # print(e_feature.shape, h3.shape)
        # h_e = torch.cat([e_feature, h3], axis=1)
        # h4 = F.relu(self.lin0(h_e))
        h4 = F.relu(self.lin0(h3))
        h5 = F.relu(self.lin1(h4))
        logits_gnn = self.classifier(h5)
        logits_ssl = self.discriminator(h2).squeeze()
        return logits_gnn, logits_ssl
        
    def training_step(self, batch, batch_idx):
        logits_gnn, logits_ssl = self.forward(batch)
        e_labels = batch['edge_label'].long()
        n_labels = batch['y'].float()
        ce_loss = self.loss(logits_gnn, e_labels)        
        ssl_loss = self.loss_ssl(torch.sigmoid(logits_ssl), n_labels)
        train_loss = ce_loss + ssl_loss
        logs = {
            'train_loss': train_loss,
            'ce_loss': ce_loss,
            'ssl_loss': ssl_loss,
            'batch_size': torch.tensor(batch.edge_label.shape[0], dtype=torch.float32),
        }
        self.log_dict(logs, prog_bar=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        logits_gnn, logits_ssl = self.forward(batch)
        e_labels = batch['edge_label'].long()
        n_labels = batch['y'].float()
        ce_loss = self.loss(logits_gnn, e_labels)        
        ssl_loss = self.loss_ssl(torch.sigmoid(logits_ssl), n_labels)
        total_loss = ce_loss + ssl_loss
        acc, f1_macro, f1_micro = utils.calc_metrics(logits_gnn, e_labels, num_labels=self.hparams.num_labels)
        logs = {
            'total_loss': total_loss,
            'ce_loss': ce_loss, 
            'acc': acc,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'batch_size': torch.tensor(batch.edge_label.shape[0], dtype=torch.float32),
        }
        self.log_dict(logs, prog_bar=True)
        return logs
    
    def validation_epoch_end(self, val_step_outputs):
        avg_loss = torch.stack([x['total_loss'] for x in val_step_outputs]).mean()
        avg_ce_loss = torch.stack([x['ce_loss'] for x in val_step_outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in val_step_outputs]).mean()
        avg_f1_macro = torch.stack([x['f1_macro'] for x in val_step_outputs]).mean()
        avg_f1_micro = torch.stack([x['f1_micro'] for x in val_step_outputs]).mean()
        logs = {
            'val_loss': avg_loss,
            'val_ce_loss': avg_ce_loss, 
            'val_acc': avg_acc,
            'val_macro_f1': avg_f1_macro,
            'val_micro_f1': avg_f1_micro,
        }
        self.log_dict(logs, prog_bar=True)
     
    def test_step(self, batch, batch_idx):
        logits_gnn, logits_ssl = self.forward(batch)
        e_labels = batch['edge_label'].long()
        n_labels = batch['y'].float()
        ce_loss = self.loss(logits_gnn, e_labels)        
        ssl_loss = self.loss_ssl(torch.sigmoid(logits_ssl), n_labels)
        total_loss = ce_loss + ssl_loss
        acc, f1_macro, f1_micro = utils.calc_metrics(logits_gnn, e_labels, num_labels=self.hparams.num_labels)
        logs = {
            'total_loss': total_loss,
            'ce_loss': ce_loss, 
            'acc': acc,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'batch_size': torch.tensor(batch.edge_label.shape[0], dtype=torch.float32),
        }
        return logs
    
    def test_epoch_end(self, test_step_outputs):
        avg_loss = torch.stack([x['total_loss'] for x in test_step_outputs]).mean()
        avg_ce_loss = torch.stack([x['ce_loss'] for x in test_step_outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in test_step_outputs]).mean()
        avg_f1_macro = torch.stack([x['f1_macro'] for x in val_step_outputs]).mean()
        avg_f1_micro = torch.stack([x['f1_micro'] for x in val_step_outputs]).mean()
        logs = {
            'test_loss': avg_loss,
            'test_ce_loss': avg_ce_loss, 
            'test_acc': avg_acc,
            'test_macro_f1': avg_f1_macro,
            'test_micro_f1': avg_f1_micro,
        }
        self.log_dict(logs, prog_bar=True)
        return logs
   
    def predict(self, data):
        e_feature = data.edge_attr
        h2 = F.relu(self.gnn(data.x, data.edge_index, data.edge_attr))
        h3 = self.edge_encoder(h2, data.edge_label_index)
        # print(e_feature.shape, h3.shape)
        # h_e = torch.cat([e_feature, h3], axis=1)
        # h4 = F.relu(self.lin0(h_e))
        h4 = F.relu(self.lin0(h3))
        h5 = F.relu(self.lin1(h4))
        logits_gnn = self.classifier(h5)
        probs = F.softmax(logits_gnn, 1)
        return probs