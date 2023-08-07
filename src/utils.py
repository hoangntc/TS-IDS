import os, sys, re, datetime, random, gzip, json
import commentjson
from collections import OrderedDict
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import accumulate
import argparse
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F

def read_json(fname):
    '''
    Read in the json file specified by 'fname'
    '''
    with open(fname, 'rt') as handle:
        return commentjson.load(handle, object_hook=OrderedDict)

def read_data(data_config, returned_dtype='tensor', verbose=True):
    assert returned_dtype in ['tensor', 'array']
    fname = os.path.join(data_config['root'], data_config['ds_name']+'.pkl')
    print(fname)
    g_data = pd.read_pickle(fname)
    x = g_data['n_features']
    edge_index = g_data['edge_index']
    edge_attr = g_data['e_features']
    y = g_data['node_label']
    input_train_edges = g_data['edge_index'][:, np.where(g_data['tvt']=='train')[0]]
    input_val_edges = g_data['edge_index'][:, np.where(g_data['tvt']=='val')[0]]
    input_test_edges = g_data['edge_index'][:, np.where(g_data['tvt']=='test')[0]]
    
    input_train_labels = g_data['edge_label'][np.where(g_data['tvt']=='train')[0]]
    input_val_labels = g_data['edge_label'][np.where(g_data['tvt']=='val')[0]]
    input_test_labels = g_data['edge_label'][np.where(g_data['tvt']=='test')[0]]
    
    input_train_edges_attr = g_data['e_features'][np.where(g_data['tvt']=='train')[0]]
    input_val_edges_attr = g_data['e_features'][np.where(g_data['tvt']=='val')[0]]
    input_test_edges_attr = g_data['e_features'][np.where(g_data['tvt']=='test')[0]]
    if returned_dtype == 'tensor':
        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        input_train_edges = torch.tensor(input_train_edges, dtype=torch.long)
        input_val_edges = torch.tensor(input_val_edges, dtype=torch.long)
        input_test_edges = torch.tensor(input_test_edges, dtype=torch.long)
        input_train_labels = torch.tensor(input_train_labels, dtype=torch.long)
        input_val_labels = torch.tensor(input_val_labels, dtype=torch.long)
        input_test_labels = torch.tensor(input_test_labels, dtype=torch.long)
        input_train_edges_attr = torch.tensor(input_train_edges_attr, dtype=torch.long)
        input_val_edges_attr = torch.tensor(input_val_edges_attr, dtype=torch.long)
        input_test_edges_attr = torch.tensor(input_test_edges_attr, dtype=torch.long)
    if verbose:
        print('x:', x.shape)
        print('edge_index:', edge_index.shape)
        print('edge_attr:', edge_attr.shape)
        print('y:', y.shape)
        print('input_train_edges:', input_train_edges.shape)
        print('input_val_edges:', input_val_edges.shape)
        print('input_test_edges:', input_test_edges.shape)
        print('input_train_labels:', input_train_labels.shape)
        print('input_val_labels:', input_val_labels.shape)
        print('input_test_labels:', input_test_labels.shape) 
    return g_data, x, edge_index, edge_attr, y, input_train_edges, input_val_edges, input_test_edges, input_train_labels, input_val_labels, input_test_labels, input_train_edges_attr, input_val_edges_attr, input_test_edges_attr

def calc_auc(logits, gts, num_labels, avg_type='macro', binary=False):
    if binary:
        auc = roc_auc_score(gts.cpu().detach(), F.softmax(logits, dim=1).cpu().detach())
    else: 
        auc = roc_auc_score(
            gts.cpu().detach(), 
            F.softmax(logits, dim=1).cpu().detach(), 
            average=avg_type, 
            multi_class='ovo', 
            labels=np.arange(num_labels),
        )
    return torch.tensor([auc])

def calc_f1(logits, gts, avg_type='macro', binary=False):
    '''
    Calculates the F1 score (either macro or micro as defined by 'avg_type') for the specified logits and ground truths
    '''
    if binary:
        probs = torch.sigmoid(logits)
        thresh = torch.tensor([0.5]).to(probs.device)
        pred = (probs > thresh)
    else:
        pred = torch.argmax(logits, dim=-1)
    score = f1_score(gts.cpu().detach(), pred.cpu().detach(), average=avg_type, zero_division=0)
    return torch.tensor([score])

def calc_accuracy(logits, gts, binary=False):
    '''
    Calculates the accuracy for the specified logits and gts
    '''
    if binary:
        probs = torch.sigmoid(logits)
        thresh = torch.tensor([0.5]).to(probs.device)
        pred = (probs > thresh)
    else:
        pred = torch.argmax(logits, 1)
    acc = accuracy_score(gts.cpu().detach(), pred.cpu().detach())
    return torch.tensor([acc])

def calc_metrics(logits, gts, num_labels, binary=False):    
    acc = calc_accuracy(logits, gts, binary=binary)
    f1_macro = calc_f1(logits, gts, avg_type='macro', binary=binary)
    f1_weighted = calc_f1(logits, gts, avg_type='weighted', binary=binary)
#     auc = calc_auc(logits, gts, num_labels=num_labels, avg_type=avg_type, binary=binary)
    return acc, f1_macro, f1_weighted
