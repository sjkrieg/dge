import argparse
import gc
import stellargraph as sg
import networkx as nx
import numpy as np
import os
import pandas as pd
import uuid
import tensorflow as tf
from datetime import datetime
from itertools import chain, product
from math import exp, ceil
from sgmods import *
from utils import *
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, average_precision_score, log_loss, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import optimizers, losses, metrics, Model
from tensorflow.keras.layers import Dense, Concatenate, Average
from time import perf_counter

parser = argparse.ArgumentParser()
parser.add_argument('infprefix', help='path and prefix for input graph; reads from {infprefix}{k}.txt')
parser.add_argument('-task', '--task', type=int, choices=[0], default=0, help='0 for node classification')
parser.add_argument('-k', '--k', type=int, default=2, help='max order for input graph')
parser.add_argument('-l', '--learners', type=int, default=16, help='number of base GNNs')
parser.add_argument('-p', '--pool', type=int, choices=[0,1,2,3], default=2, help='0 for dge-concat (eq5a), 1 for dge-pool (eq5b)), 2 for dge-bag (eq5c), 3 for dge-batch*')
parser.add_argument('-q', '--shareparams', action='store_true', default=False)
parser.add_argument('-n', '--neighborsamples', nargs='+', type=int, default=[128,1])
parser.add_argument('-a', '--aggregator', type=int, choices=[0,1,2], default=0, help='0=GraphSAGE mean, 1=GIN (sum), 2=GraphSAGE maxpool')
parser.add_argument('-s', '--seed', type=int, default=7)
parser.add_argument('-f', '--testfold', type=int, default=0, choices=[0,1,2,3,4])
parser.add_argument('-d', '--hiddendims', nargs='+', type=int, default=[128,128])
parser.add_argument('-x', '--dropout', type=float, default=0.4)
parser.add_argument('-b', '--batchsize', type=int, default=16)
parser.add_argument('-e', '--epochs', type=int, default=25)
parser.add_argument('-r', '--learningrate', type=float, default=0.01)
parser.add_argument('-t', '--tau', type=int, default=0)
parser.add_argument('-o', '--outdir', default='results/')
args = vars(parser.parse_args())

# sanity checks
if len(args['neighborsamples']) != len(args['hiddendims']):
    raise ValueError('Expected argument {} to have {} values (passed value was {})'.format(curr, len(args['hiddendims']), args['neighborsamples']))
# pool = 3 only supports shared parameters
if args['pool'] == 3: args['shareparams'] = True
# use pool = 0 to include validation results during training for baselines
if args['learners'] == 1: args['pool'] = 0
print('\nrunning with the following configurations:')
for key, val in args.items():
    print('--> {:>16} = {}'.format(key, val))

# loading graphs and general preprocessing
hon_suffix = ''
if args['tau']: hon_suffix += 't{:02d}'.format(args['tau'])
inf_name = args['infprefix'] + '{}{}.txt'.format(args['k'], hon_suffix if args['k'] > 1 else '')
print('\nreading input graph from {}...'.format(inf_name), end='')
N, E, X, N_k1, N_k1_names, n2fam, n2id, id2n = load_graph(inf_name)
print('done!')

# node classification
if args['task'] == 0:
    print('preparing for node classification task!')
    print('converting to StellarGraph objects...', end='')
    # known issue with float16 conversion -- https://github.com/scipy/scipy/issues/7408
    G = StellarGraphMod(X.toarray().astype(dtype), E, is_directed=True)
    del X # free some memory
    gc.collect()
    print('done!\n')
    print(G.info())
    
    print('generating labels and train/test splits...', end='')
    y, n_classes = load_labels(args['infprefix'] + '_labels.csv', N_k1_names, n2id)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=args['seed']).split(N_k1, y.idxmax(axis=1)))
    print('done!')
    train_idx, test_idx = folds[args['testfold']]
    print('\ntrain nodes:',train_idx.shape)
    print('train label distribution:')
    print(y.loc[train_idx].idxmax(axis=1).value_counts(dropna=False, normalize=True).sort_index())
    print('\ntest nodes:',test_idx.shape)
    print('test label distribution:')
    print(y.loc[test_idx].idxmax(axis=1).value_counts(dropna=False, normalize=True).sort_index())
    
    # prepare for model initialization
    if n_classes > 2:
        loss = losses.categorical_crossentropy
        metrics = ["acc"]
    else:
        loss = losses.binary_crossentropy
        metrics = ["acc"]
    aggregator = GINAggregator if args['aggregator'] == 1 else sg.layer.MaxPoolingAggregator if args['aggregator'] == 2 else sg.layer.MeanAggregator
    rng = np.random.default_rng(args['seed'])
    
    # equation 5a/5b -- dge-concat/dge-pool
    if args['pool'] in [0,1]:
        print('initializing node generator and generating relative bootstraps...', end='')
        gen = DGENodeGenerator(G, args['learners'], args['batchsize'], args['neighborsamples'], args['neighborsamples'], weighted=True, n2fam=n2fam, N_k1=N_k1)
        train_gen = gen.flow(train_idx, y.loc[train_idx], shuffle=True)
        test_gen = gen.flow(test_idx, y.loc[test_idx])
        print('done!')
        
        print('instantiating model...', end='')
        # share parameters by duplicating the graphsage layer
        if args['shareparams']:
            sages = [DirectedGraphSAGEMod(layer_sizes=args['hiddendims'], generator=gen, normalize='l2', bias=True, dropout=args['dropout'], aggregator=aggregator)] * args['learners']
        else:
            sages = [DirectedGraphSAGEMod(layer_sizes=args['hiddendims'], generator=gen, normalize='l2', bias=True, dropout=args['dropout'], aggregator=aggregator) for _ in range(args['learners'])]
        sage_ios = [sage.in_out_tensors() for sage in sages]
        sage_inputs = [io[0] for io in sage_ios]
        sage_outputs = [io[1] for io in sage_ios]
    
        if len(sage_outputs) > 1:
            if args['pool'] == 0:
                pool = Concatenate()(sage_outputs)
            elif args['pool'] == 1:
                pool = Average()(sage_outputs)
        else: # only if running graphsage or gin as baselines
            pool = sage_outputs[0]
            
        if n_classes > 2:
            prediction = Dense(units=n_classes, activation='softmax')(pool)
        else:
            prediction = Dense(units=1, activation='sigmoid')(pool)
            
        model = Model(inputs=sage_inputs, outputs=prediction)
        
        model.compile(
            optimizer=optimizers.Adam(lr=args['learningrate']),
            loss=loss,
            metrics=metrics
        )
        print(model.summary())
        print('beginning training!')
        history = model.fit(train_gen, epochs=args['epochs'], verbose=1, shuffle=False, validation_data=test_gen)
        model_out = model.predict(test_gen)
        param_count = model.count_params()
    
    # equation 5c / dge-bag
    elif args['pool'] == 2:
        print('sampling relatives...', end='')
        relative_samples_train, relative_samples_test = [sample_relatives(idx, args['learners'], n2fam, rng) for idx in [train_idx, test_idx]]
        print('done!')
        
        model, model_outs = None, []
        gen = HONGraphSAGENodeGenerator(G, args['batchsize'], args['neighborsamples'], args['neighborsamples'], seed=args['seed'], weighted=True)
        for i in range(args['learners']):
            print('\n***training learner {} / {}...'.format(i+1, args['learners']))
            curr_train_idx, curr_test_idx = relative_samples_train[i], relative_samples_test[i]
            # TODO: add teleports for ensemble training
            train_gen = gen.flow(curr_train_idx, y.loc[train_idx], shuffle=True)
            test_gen = gen.flow(curr_test_idx, y.loc[test_idx], shuffle=False)

            if not args['shareparams'] or not model:
                print('instantiating new learner...')
                sage = DirectedGraphSAGEMod(layer_sizes=args['hiddendims'], generator=gen, normalize='l2', bias=True, dropout=args['dropout'], aggregator=aggregator)
                sage_input, sage_output = sage.in_out_tensors()
                if n_classes > 2:
                    prediction = Dense(units=n_classes, activation='softmax')(sage_output)
                else:
                    prediction = Dense(units=1, activation='sigmoid')(sage_output)
                
                model = Model(inputs=sage_input, outputs=prediction)
                model.compile(
                    optimizer=optimizers.Adam(lr=args['learningrate']),
                    loss=loss,
                    metrics=metrics
                )
            else:
                print('recycling parameters from previous learner...')
            if not i: print(model.summary())
            
            history = model.fit(train_gen, epochs=args['epochs'], verbose=1, shuffle=False)
            model_outs.append(model.predict(test_gen))
            
            print('first {} learners: '.format(i+1), end='')
            model_out = np.mean(np.stack(model_outs, axis=-1), axis=-1)
            if n_classes == 2:
                y_true_classes = y.loc[test_idx]
                y_pred_classes = np.rint(model_out).astype(int)
            else:
                y_true_classes = y.loc[test_idx].idxmax(axis=1)
                y_pred_classes = model_out.argmax(axis=-1)
            print('{:.2f} / {:.2f} (val f1 / loss)'.format(f1_score(y_true_classes, y_pred_classes, average='micro'), log_loss(y.loc[test_idx], model_out)))
        
        # generate final outputs
        if len(model_outs) > 1:
            model_out = np.mean(np.stack(model_outs, axis=-1), axis=-1)
        else:
            model_out = model_outs[0]
        param_count = model.count_params() * args['learners']
    
    # equation 5c for inference, but minibatch relative sampling (dge-batch*)
    elif args['pool'] == 3:
        print('sampling relatives...', end='')
        # resample relatives prior to training
        relative_samples_train, relative_samples_test = [sample_relatives(idx, l, n2fam, rng) for l, idx in zip([args['epochs'], args['learners']], [train_idx, test_idx])]
        print('done!')
        
        model_outs = []
        gen = HONGraphSAGENodeGenerator(G, args['batchsize'], args['neighborsamples'], args['neighborsamples'], seed=args['seed'], weighted=True)
        
        sage = DirectedGraphSAGEMod(layer_sizes=args['hiddendims'], generator=gen, normalize='l2', bias=True, dropout=args['dropout'], aggregator=aggregator)
        sage_input, sage_output = sage.in_out_tensors()
        if n_classes > 2:
            prediction = Dense(units=n_classes, activation='softmax')(sage_output)
        else:
            prediction = Dense(units=1, activation='sigmoid')(sage_output)
        
        model = Model(inputs=sage_input, outputs=prediction)
        model.compile(
            optimizer=optimizers.Adam(lr=args['learningrate']),
            loss=loss,
            metrics=metrics
        )
        print(model.summary())
        
        for i in range(args['epochs']):
            print('***training epoch {} / {}...'.format(i+1, args['epochs']))
            train_gen = gen.flow(relative_samples_train[i], y.loc[train_idx], shuffle=True)
            history = model.fit(train_gen, epochs=1, verbose=1, shuffle=False)
        
        model_outs = []
        for i in range(args['learners']):
            test_gen = gen.flow(relative_samples_test[i], y.loc[test_idx], shuffle=False)
            model_outs.append(model.predict(test_gen))
            print('first {} readouts: '.format(i+1), end='')
            model_out = np.mean(np.stack(model_outs, axis=-1), axis=-1)
            if n_classes == 2:
                y_true_classes = y.loc[test_idx]
                y_pred_classes = np.rint(model_out).astype(int)
            else:
                y_true_classes = y.loc[test_idx].idxmax(axis=1)
                y_pred_classes = model_out.argmax(axis=-1)
            print('{:.2f} / {:.2f} (val f1 / loss)'.format(f1_score(y_true_classes, y_pred_classes, average='micro'), log_loss(y.loc[test_idx], model_out)))
        
        # generate final outputs
        if len(model_outs) > 1:
            model_out = np.mean(np.stack(model_outs, axis=-1), axis=-1)
        else:
            model_out = model_outs[0]
        param_count = model.count_params() * args['learners']
    # args['ensemblemode'] is 0

    log = pd.DataFrame.from_dict(args, orient='index').transpose()
    log['completed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for metric in history.history.keys():
        log[metric] = history.history[metric][-1]
    
    if n_classes > 2:
        y_true_classes = y.loc[test_idx].idxmax(axis=1)
        y_pred_classes = model_out.argmax(axis=-1)
    else:
        y_true_classes = y.loc[test_idx]
        y_pred_classes = np.rint(model_out).astype(int)
        
    log['val_f1'] = f1_score(y_true_classes, y_pred_classes, average='micro')
    log['val_loss'] = log_loss(y.loc[test_idx], model_out)
    log['modelsize'] = param_count
    print('f1 score {}'.format(log['val_f1']))
    print('val loss {}'.format(log['val_loss']))

otf_id = str(uuid.uuid4())
if not os.path.isdir(args['outdir']):
    os.mkdir(args['outdir'])
otf_name = args['outdir'] + otf_id + '.csv'
print('writing results to {}...'.format(otf_name))
log.to_csv(otf_name, index=False)
print('done!')