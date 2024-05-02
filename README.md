# TGN: Temporal Graph Networks [[arXiv](https://arxiv.org/abs/2006.10637), [YouTube](https://www.youtube.com/watch?v=W1GvX2ZcUmY), [Blog Post](https://towardsdatascience.com/temporal-graph-networks-ab8f327f2efe)] 

Dynamic Graph             |  TGN	
:-------------------------:|:-------------------------:	
![](figures/dynamic_graph.png)  |  ![](figures/tgn.png)	

#### Paper link: [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637)


## What's Difference between tgn and tgn-aa
Multiple choices can be condidered for implementing the Message Aggregator module. While the original paper only considered two efficient __non-learnable__ solutions: most recent message (keep only most recent message for a given node) and mean message (average all messages for a given node), our tgn-aa design a __learable attention-based aggregation__ to aggregate the messages from multiple events for nodes in the same batch


## What's change
```lua
|-modules
|- |-message_aggregator.py add AttentionMessageAggregator
|-train_self_supervised.py add attention for argument --aggregator
|-train_supervised.py add attention for argument --aggregator
```

### Preprocess the data
We use the dense `npy` format to save the features in binary format. If edge features or nodes 
features are absent, they will be replaced by a vector of zeros. 
```{bash}
python utils/preprocess_data.py --data wikipedia --bipartite
```

### Training on Wikipedia Dataset
```{bash}
### tgn-aa
# TGN-attn with attention aggregator: Self-Supervised learning on the wikipedia dataset
## unlearnable
python train_self_supervised.py --aggregator attention --use_memory --prefix tgn-attn --n_runs 10
## learnable
python train_self_supervised.py --aggregator attention --learnable --use_memory --prefix tgn-attn --n_runs 10

# TGN-attn with attention aggregator: supervised learning on the wikipedia dataset
## unlearnable
python train_supervised.py --aggregator attention --use_memory --prefix tgn-attn --n_runs 10
## learnable
python train_supervised.py --aggregator attention --learnable --use_memory --prefix tgn-attn --n_runs 10

### Baselines
# Jodie
python train_self_supervised.py --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --n_runs 10

# Jodie
python train_supervised.py --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --n_runs 10
```



