#defining the simple job
import os
import itertools
import numpy as np
import pandas as pd
import yaml
import logging


with open('config.yaml','r') as fid:
    config=yaml.safe_load(fid)

try:
    with open('config.yaml','r') as fid:
        configuration = yaml.safe_load(fid)
except:
    from config import configuration

# Start tree_maker logging if log_file is present in config
try:
    import tree_maker
    if 'log_file' not in configuration.keys():
        tree_maker=None
except:
    tree_maker=None

if tree_maker is not None:
    tree_maker.tag_json.tag_it(configuration['log_file'], 'started')





n_part = config['n_part']
x_in_sigmas = np.random.normal(0, 1, n_part)
px_in_sigmas = np.random.normal(0, 1, n_part)
y_in_sigmas = np.random.normal(0, 1, n_part)
py_in_sigmas = np.random.normal(0, 1, n_part)

#produce a dataframe with x_in_sigmas,px_in_sigmas,y_in_sigmas,py_in_sigmas
df = pd.DataFrame({'x_in_sigmas': x_in_sigmas, 'px_in_sigmas': px_in_sigmas, 'y_in_sigmas': y_in_sigmas, 'py_in_sigmas': py_in_sigmas})
#save the dataframe to a parquet file
df.to_parquet('transverse_distr.parquet')

if tree_maker is not None:
    tree_maker.tag_json.tag_it(config['log_file'], 'completed')