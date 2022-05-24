import math
import pandas as pd
import stellargraph as sg
import tensorflow as tf
from collections import defaultdict
from itertools import chain
from stellargraph import StellarGraph
from stellargraph.data.explorer import DirectedBreadthFirstNeighbours
from stellargraph.mapper import DirectedGraphSAGENodeGenerator, DirectedGraphSAGELinkGenerator
from stellargraph.layer import DirectedGraphSAGE, GCN, GraphConvolution, GAT, GraphAttentionSparse
from stellargraph.layer.misc import SqueezedSparseConversion, deprecated_model_function, GatherIndices
from stellargraph.layer.graphsage import GraphSAGEAggregator
from tensorflow.keras.layers import Input, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from time import perf_counter
from utils import *
import itertools as it
import numpy as np

# overrode this class to resample relatives within the generator class
class DGENodeGenerator(DirectedGraphSAGENodeGenerator):
    def __init__(self,
        G,
        n_learners,
        batch_size,
        in_samples,
        out_samples,
        seed=None,
        name=None,
        weighted=False,
        n2fam=None,
        N_k1=None
    ):

        self.rng = np.random.default_rng(seed)
        super().__init__(G, batch_size, in_samples, out_samples, seed, name, weighted)
        
        self.n_learners = n_learners
        self.n2fam = n2fam

        # stuff to override
        self.relative_samples = sample_relatives(N_k1, self.n_learners, self.n2fam, self.rng)
        self.generators = [HONGraphSAGENodeGenerator(G, batch_size, in_samples, out_samples, seed=seed, weighted=weighted) for _ in range(self.n_learners)]
        
    def sample_features(self, head_nodes, batch_num):
        return [gen.sample_features(self.relative_samples[i][head_nodes], batch_num) for i, gen in enumerate(self.generators)]
        

# overrode this class to implement the sparse BFS neighbor sampler
class HONGraphSAGENodeGenerator(DirectedGraphSAGENodeGenerator):
    def __init__(self,
        G,
        batch_size,
        in_samples,
        out_samples,
        seed=None,
        name=None,
        weighted=False,
    ):
        super().__init__(G, batch_size, in_samples, out_samples, seed, name, weighted)
        
        self.sampler = DirectedBreadthFirstNeighboursSparse(
            G, graph_schema=self.schema, seed=seed
        )

# overrode this method to prevent recasting features to a different dtype
def extract_element_features_mod(element_data, unique, name, ids, type, use_ilocs):
    if ids is None:
        if type is None:
            type = unique(
                f"{name}_type: in a non-homogeneous graph, expected a {name} type and/or '{name}s' to be passed; found neither '{name}_type' nor '{name}s', and the graph has {name} types: %(found)s"
            )

        return element_data.features_of_type(type)

    ids = np.asarray(ids)

    if len(ids) == 0:
        # empty lists are cast to a default array type of float64 -
        # must manually specify integer type if empty, in which case we can pretend we received ilocs
        ilocs = ids.astype(dtype=np.float16)
        use_ilocs = True
    elif use_ilocs:
        ilocs = ids
    else:
        ilocs = element_data.ids.to_iloc(ids)

    valid = element_data.ids.is_valid(ilocs)
    all_valid = valid.all()
    valid_ilocs = ilocs if all_valid else ilocs[valid]

    if type is None:
        try:
            # no inference required in a homogeneous-node graph
            type = unique()
        except ValueError:
            # infer the type based on the valid nodes
            types = np.unique(element_data.type_of_iloc(valid_ilocs))

            if len(types) == 0:
                raise ValueError(
                    f"must have at least one node for inference, if `{name}_type` is not specified"
                )
            if len(types) > 1:
                raise ValueError(f"all {name}s must have the same type")

            type = types[0]

    if all_valid:
        return element_data.features(type, valid_ilocs)

    # If there's some invalid values, they get replaced by zeros; this is designed to allow
    # models that build fixed-size structures (e.g. GraphSAGE) based on neighbours to fill out
    # missing neighbours with zeros automatically, using None as a sentinel.

    # FIXME: None as a sentinel forces nodes to have dtype=object even with integer IDs, could
    # instead use an impossible integer (e.g. 2**64 - 1)

    # everything that's not the sentinel should be valid
    if not use_ilocs:
        non_nones = ids != None
        element_data.ids.require_valid(ids[non_nones], ilocs[non_nones])

    sampled = element_data.features(type, valid_ilocs)
    features = np.zeros((len(ids), sampled.shape[1]), dtype=sampled.dtype)
    features[valid] = sampled

    return features

# modified this class to make feature sampling more memory efficient
class StellarGraphMod(StellarGraph):
    def node_features(self, nodes=None, node_type=None, use_ilocs=False):
        return extract_element_features_mod(
            self._nodes, self.unique_node_type, "node", nodes, node_type, use_ilocs
        )
        
# modifying directedgraphsage only to specify input dtype
class DirectedGraphSAGEMod(DirectedGraphSAGE):
    def _node_model(self):
        """
        Builds a GraphSAGE model for node prediction

        Returns:
            tuple: (x_inp, x_out) where ``x_inp`` is a list of Keras input tensors
            for the specified GraphSAGE model and ``x_out`` is the Keras tensor
            for the GraphSAGE model output.

        """
        # Create tensor inputs for neighbourhood sampling
        x_inp = [
            Input(shape=(s, self.input_feature_size), dtype=tf.float16) for s in self.neighbourhood_sizes
        ]

        # Output from GraphSAGE model
        x_out = self(x_inp)

        # Returns inputs and outputs
        return x_inp, x_out

class SparseSampler:
    def __init__(self, n, p, s=2**10):
        self.n = np.reshape(n, (-1, 1))
        if p.sum() != 1.0:
            self.p = p / p.sum()
        else:
            self.p = p
        self.size = s
        self.rng = np.random.default_rng(7)
        self.__repopulate__()
    
    # remaining len
    def __len__(self):
        return len(self.q) - i
    
    def __iter__(self):
        return self
    
    def __repopulate__(self):
        self.q = self.rng.choice(self.n, size=self.size, p=self.p, replace=True)
        self.i = -1
        
    def __next__(self):
        # this will skip index 0 but oh well
        self.i += 1
        try:
            return self.q[self.i]
        except IndexError:
            self.__repopulate__()
            return self.__next__()


class DirectedBreadthFirstNeighboursSparse(DirectedBreadthFirstNeighbours):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # indexed by node to SparseSampler class
        self.sparse_samplers = defaultdict(dict)
        
    # override the main sampling function to be more efficient with sparse cases
    def _sample_neighbours_untyped(
        self, neigh_func, py_and_np_rs, cur_node, size, weighted
    ):
        """
        Sample ``size`` neighbours of ``cur_node`` without checking node types or edge types, optionally
        using edge weights.
        """
        if cur_node != -1:
            neighbours = neigh_func(
                cur_node, use_ilocs=True, include_edge_weight=weighted
            )

            if weighted:
                neighbours, weights = neighbours
        else:
            neighbours = []

        if len(neighbours) > 0:
            if weighted:
                # sample following the edge weights
                if size > 1:
                    idx = sg.data.explorer.naive_weighted_choices(py_and_np_rs[1], weights, size=size)
                else:
                    try:
                        idx = next(self.sparse_samplers[neigh_func.__name__][cur_node])
                    except KeyError:
                        self.sparse_samplers[neigh_func.__name__][cur_node] = SparseSampler(np.array(range(len(neighbours))), weights)
                        idx = next(self.sparse_samplers[neigh_func.__name__][cur_node])
                if idx is not None:
                    return neighbours[idx]
            else:
                # uniform sample; for small-to-moderate `size`s (< 100 is typical for GraphSAGE), random
                # has less overhead than np.random
                return np.array(py_and_np_rs[0].choices(neighbours, k=size))

        # no neighbours (e.g. isolated node, cur_node == -1 or all weights 0), so propagate the -1 sentinel
        return np.full(size, -1)


class GINAggregator(GraphSAGEAggregator):
    """
    Sum Aggregator for GIN implemented with Keras base layer

    Args:
        output_dim (int): Output dimension
        bias (bool): Optional bias
        act (Callable or str): name of the activation function to use (must be a
            Keras activation function), or alternatively, a TensorFlow operation.

    """
    def group_aggregate(self, x_group, group_idx=0):
        """
        Mean aggregator for tensors over the neighbourhood for each group.

        Args:
            x_group (tf.Tensor): : The input tensor representing the sampled neighbour nodes.
            group_idx (int, optional): Group index.

        Returns:
            :class:`tensorflow.Tensor`: A tensor aggregation of the input nodes features.
        """
        # The first group is assumed to be the self-tensor and we do not aggregate over it
        if group_idx == 0:
            x_agg = x_group
        else:
            x_agg = K.sum(x_group, axis=2)

        return K.dot(x_agg, self.w_group[group_idx])