from collections import defaultdict
from typing import Optional
import torch
import numpy as np
import torch.nn as nn


class MessageAggregator(torch.nn.Module):
  """
  Abstract class for the message aggregator module, which given a batch of node ids and
  corresponding messages, aggregates messages with the same node id.
  """
  def __init__(self, device):
    super(MessageAggregator, self).__init__()
    self.device = device

  def aggregate(self, node_ids, messages):
    """
    Given a list of node ids, and a list of messages of the same length, aggregate different
    messages for the same id using one of the possible strategies.
    :param node_ids: A list of node ids of length batch_size
    :param messages: A tensor of shape [batch_size, message_length]
    :param timestamps A tensor of shape [batch_size]
    :return: A tensor of shape [n_unique_node_ids, message_length] with the aggregated messages
    """

  def group_by_id(self, node_ids, messages, timestamps):
    node_id_to_messages = defaultdict(list)

    for i, node_id in enumerate(node_ids):
      node_id_to_messages[node_id].append((messages[i], timestamps[i]))

    return node_id_to_messages


class LastMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(LastMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""    
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []
    
    to_update_node_ids = []
    
    for node_id in unique_node_ids:
        if len(messages[node_id]) > 0:
            to_update_node_ids.append(node_id)
            unique_messages.append(messages[node_id][-1][0])
            unique_timestamps.append(messages[node_id][-1][1])
    
    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class MeanMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(MeanMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    n_messages = 0

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        n_messages += len(messages[node_id])
        to_update_node_ids.append(node_id)
        unique_messages.append(torch.mean(torch.stack([m[0] for m in messages[node_id]]), dim=0))
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class AttentionMessageAggregator(MessageAggregator):
  def __init__(self, device, n_heads: int, message_dim: int, dropout: float=0, post_norm: Optional[bool] = None):
    super(AttentionMessageAggregator, self).__init__(device)
    self.device = device
    self.n_heads = n_heads
    self.message_dim = message_dim
    self.q_proj = nn.Linear(in_features=message_dim, out_features=message_dim)
    self.k_proj = nn.Linear(in_features=message_dim, out_features=message_dim)
    self.v_proj = nn.Linear(in_features=message_dim, out_features=message_dim)

    self.dropout = nn.Dropout(p=dropout)
    self.layer_norm = nn.LayerNorm(normalized_shape=message_dim) if post_norm else nn.Identity()

  def attention(self, padding_message: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
    num_unique_nodes, padding_length, message_dim = padding_message.shape
    head_dimension = message_dim // self.n_heads
    if head_dimension * self.n_heads != message_dim:
      raise ValueError(f"The head number should be divisible of the dimension of the message, \
                       but got message dimension{message_dim}, and num heads {self.n_heads}")
    
    query_padding_message = self.q_proj(padding_message).view(num_unique_nodes, padding_length, self.n_heads, head_dimension)
    key_padding_message = self.k_proj(padding_message).view(num_unique_nodes, padding_length, self.n_heads, head_dimension)
    value_padding_message = self.v_proj(padding_message).view(num_unique_nodes, padding_length, self.n_heads, head_dimension)

    query_padding_message = query_padding_message.permute(0, 2, 1, 3)
    key_padding_message = key_padding_message.permute(0, 2, 3, 1)
    value_padding_message = value_padding_message.permute(0, 2, 1, 3)
    scale = 1 / torch.sqrt(torch.tensor(message_dim))
    
    attn = torch.matmul(query_padding_message, key_padding_message) * scale
    if attention_mask is not None:
      if len(attention_mask.shape) == 2:
        attention_mask = attention_mask.unsqueeze(1).tile(1, padding_length, 1) # [num_unique_nodes, padding_length, padding_length]
        attention_mask = attention_mask.unsqueeze(1).tile(1, self.n_heads, 1, 1) # [num_unique_nodes, num_heads, padding_length, padding_length]
      else:
        raise ValueError("Shape of attention mask should be [num_unique_nodes, padding_length], other shape currently not support")
      attn = attn.masked_fill(attention_mask==1, -1e9)
      attn = self.dropout(attn)

    output = torch.matmul(attn, value_padding_message).permute(0, 2, 1, 3) # [num_unique_nodes, padding_length, self.n_heads, head_dimension]
    output = output.reshape(num_unique_nodes, -1, message_dim)
    output = self.layer_norm(output)
    return output
  
  def aggregate(self, node_ids, messages):
    '''
    use attention mechanism to aggregate the message for nodes in the batch
    '''
    unique_node_ids = np.unique(node_ids)
    message_dim = self.message_dim
    unique_timestamps = []
    unique_messages = []
    to_update_node_ids = []

    length_message = [len(messages[node_id]) for node_id in unique_node_ids]
    max_length = max(length_message)
    num_unique_nodes = np.count_nonzero(np.array(length_message))
    
    if max_length == 0:
      return to_update_node_ids, unique_messages, unique_timestamps
    else:
      # generate attention mask and padding_message
      attention_mask = torch.zeros(num_unique_nodes, max_length, device=self.device)
      padding_message = torch.zeros(num_unique_nodes, max_length, message_dim, device=self.device)
      idx = 0
      for node_id in unique_node_ids:
        if len(messages[node_id]) > 0:
          to_update_node_ids.append(node_id)
          length_per_message = len(messages[node_id])
          attention_mask[idx, length_per_message:] = 1
          padding_message[idx, range(length_per_message)] = torch.stack([m[0] for m in messages[node_id]], dim=0)
          unique_timestamps.append(messages[node_id][-1][1])
          idx += 1
        
      cls_message = torch.zeros(num_unique_nodes, 1, message_dim, device=self.device)
      cls_attention_mask = torch.zeros(num_unique_nodes, 1, device=self.device)
      padding_message_with_cls_token = torch.concat((cls_message, padding_message), dim=1)
      attention_mask_with_cls_token = torch.concat((cls_attention_mask, attention_mask), dim=1)
    
      updated_attention_message = self.attention(padding_message_with_cls_token, attention_mask_with_cls_token) # [num_unique_nodes, max_length, message_dim]
    
      unique_messages = updated_attention_message[:, 0, :].squeeze(1) if len(to_update_node_ids) > 0 else []
      unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []
    
      return to_update_node_ids, unique_messages, unique_timestamps


def get_message_aggregator(aggregator_type, device, n_heads, message_dim):
  if aggregator_type == "last":
    return LastMessageAggregator(device=device)
  elif aggregator_type == "mean":
    return MeanMessageAggregator(device=device)
  elif aggregator_type == "attention":
    return AttentionMessageAggregator(device=device, n_heads=n_heads, message_dim=message_dim)
  else:
    raise ValueError("Message aggregator {} not implemented".format(aggregator_type))
