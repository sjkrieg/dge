import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--groupcols', nargs='+', default=[])
parser.add_argument('-r', '--resultcol', default='val_f1')
parser.add_argument('-o', '--otfname', default='results_all')
args = vars(parser.parse_args())

default = ['task', 'infprefix', 'k','learners', 'pool','shareparams','neighborsamples','aggregator','hiddendims','epochs','batchsize','dropout','learningrate','tau']
group_cols = default + args['groupcols']
result_col = [args['resultcol']]
#inf_dir = 'results/2022-04-15_sampscaling/'
inf_dir = 'results/'
otf_name_sum = 'results_summary_{}.csv'.format(args['resultcol'])


df = pd.concat([pd.read_csv(inf_dir + f) for f in os.listdir(inf_dir) if 'results' not in f and f.endswith('.csv')])
df.to_csv(inf_dir + args['otfname'] + '.csv', index=False)
group_cols = [g for g in group_cols if g in df.columns]

df = pd.concat([df.groupby(group_cols, dropna=False)[result_col].mean().rename(columns={'val_f1':'val_f1_mean'}), df.groupby(group_cols, dropna=False)[result_col].std().rename(columns={'val_f1':'val_f1_std'}), df.groupby(group_cols, dropna=False)[result_col].count().rename(columns={'val_f1':'val_f1_count'})], axis=1)
df.to_csv(inf_dir + otf_name_sum)
