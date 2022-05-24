# Deep graph ensembles for higher-order networks

This repository contains code for the paper "Deep Ensembles for Learning on Graphs with Higher-order Dependencies." The current version supports all node classification experiments for DGE (the proposed method), GraphSAGE, and GIN (Graph Isomorphism Network).

## Core prerequisites
1. Python 3.7.3+
2. tensorflow 2.4.1
3. StellarGraph 1.2.1
4. utils.py
5. sgmods.py

## Running an Experiment
```
python hongnn.py {INFPREFIX}
```
where {INFPREFIX} is the prefix (including folder path) for the input data set. At minimum, the code expects to find an edgelist at "{INFPREFIX}{k}.txt" for the input graph of order k. Additionally, for node classification the code expects to find a file containing node labels at "{INFPREFIX}\_labels.csv".

The results from each experiment be written to an output directory (./results/ by default). You can run "agg\_results.py" to combine the results from multiple expeirments into a single CSV.

## Optional arguments
```
-task --task (int, default=0, choices={0}):  0 for node classification, 1 for link prediction (will be supported soon).
-k --k (int, default=2): max order for input graph, this is primarily used to locate the input file.
-l --learners (int, default=16): number of base GNNs for the ensemble
-p --pool (int, default=0, choices={0,1,2,3}): 0 for dge-concat (eq. 5a), 1 for dge-pool (eq. 5b)), 2 for dge-bag (eq. 5c), 3 for dge-batch*
-q --shareparams (bool, default=False): whether base learners should share params (if pool=3 this is forced to True)
-n --neighborsamples (list of int, default=[128,1]): number of neighbors to sample at each layer (e.g., the default samples 128 one-hop neighbors, and 1 two-hop neighbor for each of the sampled one-hop neighbors)
-a --aggregator (int, default=0, choices={0,1,2}): neighborhood aggregation function; 0 for mean, 1 for sum, 2 for max
-s --seed (int, default=7): random seed
-f --testfold (int, default=0, choices={0,1,2,3,4}): current testing fold for 5-fold cross validation
-d --hiddendims (list of int, default=[128,128]): hidden layer sizes per base learner; length of this list must match length of --neighborsamples
-x --dropout (float, default=0.4): GNN dropout rate
-b --batchsize (int, default=16): minibatch size
-e --epochs (int, default=25): number of training iterations for each base learner (this should vary with your choice of --pool)
-r --learningrate (float, default=0.01): learning rate
-t --tau (int, default=0): used to specify a different HON for input
-o --outdir (str, default='results/'): output directory for experimental results
```
You can reproduce the results of the paper with the following configurations (each needs to repeat 5-fold cross validation with -f = {0,1,2,3,4}). For example,
DGE-bag:
```
python dge.py dat/air
python dge.py dat/wiki -n 64 1
```
DGE-pool*:
```
python dge.py dat/air -p 1 -q
python dge.py dat/wiki -p 1 -q -n 64 1
```
For baselines set -k to 1 and -l to 1.
GraphSAGE:
```
python dge.py dat/air -k 1 -l 1 -e 50
python dge.py dat/air -k 1 -l 1 -n 64 1 -e 50
```
GIN:
```
python dge.py dat/air -a 1 -k 1 -l 1 -e 50
python dge.py dat/air -a 1 -k 1 -l 1 -n 64 1 -e 50
```

## Using your own data
To test this code with a new data set, you'll need to provide the following inputs:
1. A network file located at {INFPREFIX}{k}.txt (e.g. dat/toy1.txt). You can use [GrowHON](https://github.com/sjkrieg/growhon) to generate these from sequence data. These must be provided in a weighted edgelist format delimited by a space. Node IDs can be strings or integers.
2. For the node classification task, a list of node labels in .csv format located at {INFPREFIX}\_labels.csv. The values in first column must match the node labels in the first order graph, e.g. toy1.txt. The .csv file must contain a column called "Label" that has a non-missing value for every node in the first-order graph. There can be any number of classes and the labels can be integers or strings.
