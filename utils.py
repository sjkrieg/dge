import numpy as np
import pandas as pd
from sgmods import *
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

dtype = np.int8 # save some memory since we only use identity features

def load_graph(inf_name):
    E = pd.read_csv(inf_name, delimiter=' ', names=['source','target','weight'], dtype={'source':str, 'target':str})
    N = pd.Series(sorted(sorted(pd.concat([E['source'], E['target']]).sort_values().unique()), key=lambda x: 1 if '|' in x else 0))
    N_k1 = N.loc[~N.str.contains('\|')].to_numpy() # need to escape pipe char

    id2n = N.to_dict() # ids to node labels
    n2id = {n:i for i,n in id2n.items()} # node labels to ids
    n2bases = N.map(lambda x: n2id[x.split('|')[0]])
    print('done!')

    print('encoding nodes and features...', end='')
    # if G1 is not a first-order graph need this to encode features
    feature_encoder = OneHotEncoder(sparse=True, dtype=dtype, handle_unknown='ignore').fit(N_k1.reshape(-1, 1))
    X = feature_encoder.transform(N.map(lambda x: str(x).split('|')[0]).astype(str).to_numpy().reshape(-1, 1))

    # generating node families (Omega_u^k)
    for ntype in ['source']:
        n_curr = E.groupby(ntype)['weight'].sum().reset_index().rename(columns={'weight':'deg'.format(ntype)})
        n_curr['base'] = n_curr[ntype].map(lambda x: str(x).split('|')[0]).map(n2id)
        n_curr['order'] = n_curr[ntype].map(lambda x: len(str(x).split('|')))
        n_curr[ntype] = n_curr[ntype].map(n2id)
        if len(n_curr['base'].unique()) < len(n_curr):
            # normalize sampling probabilities by degree
            n_curr['prob'] = n_curr.groupby('base')['deg'].transform(lambda x: x.div(x.sum())).fillna(1)
            groups = n_curr.groupby('base')
            T = pd.concat([groups[ntype].apply(list), groups['prob'].apply(list)], axis=1)
            probs = T.apply(list, axis=1).to_dict()
        else:
            probs = {}
    n2famo = probs

    # map nodes and edges to ids
    N = N.map(n2id)
    for u in ['source','target']:
        E[u] = E[u].map(n2id)
    N_k1_names = set(pd.Series(N_k1))
    N_k1 = pd.Series(N_k1).map(n2id).to_numpy()
    
    return N, E, X, N_k1, N_k1_names, n2famo, n2id, id2n

def load_labels(inf_labels, N_k1_names, n2id):
    y = pd.read_csv(inf_labels, index_col=0)['Label']
    y = y.loc[y.index.astype(str).isin(N_k1_names)] # only keep labels that have nodes in the graph
    y.index = y.index.astype(str).map(n2id)
    
    n_classes = len(y.unique())
    if n_classes > 2:
        y = pd.DataFrame(OneHotEncoder(sparse=False).fit_transform(y.to_numpy().reshape(-1,1)), index=y.index)
    return y, n_classes
    
def sample_relatives(idx, ell, n2fam, rng):
    return np.array([rng.choice(n2fam[cur_node][0], size=ell, p=n2fam[cur_node][1], replace=True) if cur_node in n2fam else np.array([cur_node] * ell, dtype=type(cur_node)) for cur_node in idx]).transpose()
