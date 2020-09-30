"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import sys
import random
import argparse
import pickle
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from model.tgn import TGN
from utils.utils import EarlyStopMonitor, get_neighbor_finder, NeighborFinder
from utils.data_processing import compute_time_statistics, get_data_node_classification

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class LR(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)



### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')


### Argument and global variables
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--tune', action='store_true',
                    help='parameters tunning mode, use train-test split on training data only.')
parser.add_argument('--new_node', action='store_true', help='model new node')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}' + '\
  node-classification.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}' + '\
  node-classification.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

full_data, node_features, edge_features, train_data, val_data, test_data = \
  get_data_node_classification(DATA, use_validation=args.tune)

max_idx = max(full_data.unique_nodes)

train_ngh_finder = get_neighbor_finder(train_data, uniform=UNIFORM, max_node_idx=max_idx)

### Model initialize
device = torch.device('cuda:{}'.format(GPU)) if torch.cuda.is_available() else "cpu"
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args.n_runs):
  results_path = "results/{}_node_classification_{}.pkl".format(args.prefix,
                                                                i) if i > 0 else "results/{}_node_classification.pkl".format(
    args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator, n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message)

  tgn = tgn.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)
  logger.debug('Num of training instances: {}'.format(num_instance))
  logger.debug('Num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)
  np.random.shuffle(idx_list)

  logger.info('Loading saved TGN model')
  model_path = f'./saved_models/{args.prefix}-{DATA}.pth'
  tgn.load_state_dict(torch.load(model_path))
  tgn.eval()
  logger.info('TGN models loaded')
  logger.info('Start training node classification task')

  lr_model = LR(node_features.shape[1], drop=DROP_OUT)
  lr_optimizer = torch.optim.Adam(lr_model.parameters(), lr=args.lr)
  lr_model = lr_model.to(device)
  idx_list = np.arange(len(train_data.sources))
  lr_criterion = torch.nn.BCELoss()
  lr_criterion_eval = torch.nn.BCELoss()


  def eval_epoch(src_l, dst_l, ts_l, label_l, batch_size, lr_model, tgan, num_layer=NODE_LAYER):
    pred_prob = np.zeros(len(src_l))
    loss = 0
    num_instance = len(src_l)
    num_batch = math.ceil(num_instance / batch_size)
    with torch.no_grad():
      lr_model.eval()
      tgan.eval()
      for k in range(num_batch):
        s_idx = k * batch_size
        e_idx = min(num_instance, s_idx + batch_size)
        src_l_cut = src_l[s_idx:e_idx]
        dst_l_cut = dst_l[s_idx:e_idx]
        ts_l_cut = ts_l[s_idx:e_idx]
        label_l_cut = label_l[s_idx:e_idx]
        size = len(src_l_cut)
        edge_idxs_batch = full_data.edge_idxs[s_idx: e_idx]
        src_embed, dst_embed, negative_embed = tgan.compute_temporal_embeddings(src_l_cut, dst_l_cut,
                                                                        dst_l_cut, ts_l_cut,
                                                                        edge_idxs_batch,
                                                                        NUM_NEIGHBORS)
        src_label = torch.from_numpy(label_l_cut).float().to(device)
        lr_prob = lr_model(src_embed).sigmoid()
        loss += lr_criterion_eval(lr_prob, src_label).item()
        pred_prob[s_idx:e_idx] = lr_prob.cpu().numpy()

    auc_roc = roc_auc_score(label_l, pred_prob)
    return auc_roc, loss / num_instance


  val_aucs = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  for epoch in range(args.n_epoch):
    # Initialize memory of the model at each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    lr_pred_prob = np.zeros(len(train_data.sources))
    np.random.shuffle(idx_list)
    tgn = tgn.eval()
    lr_model = lr_model.train()
    loss = 0
    # num_batch
    for k in range(num_batch):
      s_idx = k * BATCH_SIZE
      e_idx = min(num_instance, s_idx + BATCH_SIZE)

      sources_batch = train_data.sources[s_idx: e_idx]
      destinations_batch = train_data.destinations[s_idx: e_idx]
      timestamps_batch = train_data.timestamps[s_idx: e_idx]
      edge_idxs_batch = full_data.edge_idxs[s_idx: e_idx]
      labels_batch = train_data.labels[s_idx: e_idx]

      size = len(sources_batch)

      lr_optimizer.zero_grad()
      with torch.no_grad():
        source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                     destinations_batch,
                                                                                     destinations_batch,
                                                                                     timestamps_batch,
                                                                                     edge_idxs_batch,
                                                                                     NUM_NEIGHBORS)

      src_label = torch.from_numpy(labels_batch).float().to(device)
      lr_prob = lr_model(source_embedding).sigmoid()
      lr_loss = lr_criterion(lr_prob, src_label)
      lr_loss.backward()
      lr_optimizer.step()
      loss += lr_loss.item()
    train_losses.append(loss / num_batch)

    # train_auc, train_loss = eval_epoch(train_src_l, train_dst_l, train_ts_l, train_label_l,
    #                                  BATCH_SIZE, lr_model, tgan)
    val_auc, val_loss = eval_epoch(val_data.sources, val_data.destinations, val_data.timestamps,
                                   val_data.labels,
                                   BATCH_SIZE,
                                   lr_model, tgn)
    val_aucs.append(val_auc)

    pickle.dump({
      "val_aps": val_aucs,
      "train_losses": train_losses,
      "epoch_times": 0.0,
      "new_nodes_val_aps": [],
    }, open(results_path, "wb"))

    # torch.save(lr_model.state_dict(), './saved_models/edge_{}_wkiki_node_class.pth'.format(DATA))
    logger.info(f'train loss: {loss / num_batch}, val auc: {val_auc}')

    if early_stopper.early_stop_check(val_auc):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      break
    else:
      torch.save(lr_model.state_dict(), get_checkpoint_path(epoch))

  if args.tune:
    logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
    best_model_path = get_checkpoint_path(early_stopper.best_epoch)
    lr_model.load_state_dict(torch.load(best_model_path))
    logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
    lr_model.eval()
    test_auc, test_loss = eval_epoch(test_data.sources, test_data.destinations, test_data.timestamps,
                                   test_data.labels,
                                   BATCH_SIZE,
                                   lr_model, tgn)
  else:
    test_auc = val_aucs[-1]
  pickle.dump({
    "val_aps": val_aucs,
    "test_ap": test_auc,
    "train_losses": train_losses,
    "epoch_times": 0.0,
    "new_nodes_val_aps": [],
    "new_node_test_ap": 0,
  }, open(results_path, "wb"))

  logger.info(f'test auc: {test_auc}')