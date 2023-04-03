# %%
import tree_maker
from tree_maker import NodeJob
from tree_maker import initialize
import time
import os
from pathlib import Path
import itertools
import numpy as np
import yaml
from user_defined_functions import generate_run_sh
from user_defined_functions import generate_run_sh_htc
from utilsaf import *

# Import the configuration
config=yaml.safe_load(open('config.yaml'))



# # The user defines the variable to scan
# # machine parameters scans
# qx0 = np.arange(62.305, 62.330, 0.001)[:]
# qy0 = np.arange(60.305, 60.330, 0.001)[:]

num_turns = config['n_turns']

children={}
for idx in range(10):
    track_children={}
    children[f'{idx:03}_simulation'] = {
                                    'npart':int(20000),
                                    'children':track_children}
    track_children['000_track'] ={
        'xline_json': '../../line_bb_for_tracking.json',
        'particle_file': '../transverse_distr.parquet',
        'n_turns': num_turns            # number of turns to track
    } 


# for optics_job, (myq1, myq2) in enumerate(itertools.product(qx0, qy0)):
#     optics_children={}
#     children[study_name]["children"][f'madx_{optics_job:03}'] = {
#                                     'qx0':float(myq1),
#                                     'qy0':float(myq2),
#                                     'children':optics_children}
#     for track_job in range(15):
#         optics_children[f'xsuite_{track_job:03}'] = {
#                     'particle_file': ('../../'
#                                 f'distrib_abc/{track_job:03}.parquet'),
#                     'xline_json': ('../xsuite_lines/'
#                                 'line_bb_for_tracking.json'),
#                     'n_turns': int(1000000)}

if config['root']['use_yaml_children']== False:
    config['root']['children'] = children
# config['root']['setup_env_script'] = os.getcwd() + '/../miniconda/bin/activate'

print(f'Generating noise for {num_turns} turns')
phnoise = config['ph_noise']
anoise = config['a_noise']
ph_noise = generate_phase_noise(ph_noise_rad = phnoise, ph_noise_mu = 0,scale_noise = 1, turns = num_turns)
a_noise = generate_amplitude_noise(a_noise_mu = 0, a_noise_sigma = anoise,scale_noise = 1, turns = num_turns)
ph_noise = np.array(ph_noise)
a_noise = np.array(a_noise)
with open('noise.py', 'wb') as f:
    np.save(f,ph_noise)
    np.save(f,a_noise)


# Create tree object
start_time = time.time()
root = initialize(config)
print('Done with the tree creation.')
print("--- %s seconds ---" % (time.time() - start_time))

# From python objects we move the nodes to the file-system.
start_time = time.time()
#root.make_folders(generate_run_sh)
root.make_folders(generate_run_sh_htc)
print('The tree folders are ready.')
print("--- %s seconds ---" % (time.time() - start_time))
