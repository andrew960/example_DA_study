# ==================================================================================================
# --- Imports
# ==================================================================================================
from cpymad.madx import Madx
import os
import xmask as xm
import xmask.lhc as xlhc
import shutil
import json
import yaml
import tree_maker
import numpy as np
import itertools
import pandas as pd

# Import user-defined optics-specific tools
import optics_specific_tools_hlhc15 as ost

# ==================================================================================================
# --- Load configuration file
# ==================================================================================================
# Load configuration
with open("config.yaml", "r") as fid:
    configuration = yaml.safe_load(fid)

# Get configuration for the particles distribution and the collider separately
config_particles = configuration["config_particles"]
config_collider = configuration["config_collider"]

# Start tree_maker logging if log_file is present in config
if tree_maker is not None and "log_file" in configuration:
    tree_maker.tag_json.tag_it(configuration["log_file"], "started")

# ==================================================================================================
# --- Build particle distribution
# ==================================================================================================
# # Define radius distribution
# r_min = config_particles["r_min"]
# r_max = config_particles["r_max"]
# n_r = config_particles["n_r"]
# radial_list = np.linspace(r_min, r_max, n_r, endpoint=False)

# # Define angle distribution
# n_angles = config_particles["n_angles"]
# theta_list = np.linspace(0, 90, n_angles + 2)[1:-1]

# # Define particle distribution as a cartesian product of the above
# particle_list = [
#     (particle_id, ii[0], ii[1])
#     for particle_id, ii in enumerate(itertools.product(radial_list, theta_list))
# ]

# # Split distribution into several chunks for parallelization
# n_split = config_particles["n_split"]
# particle_list = list(np.array_split(particle_list, n_split))

# # Write distribution to parquet files
# distributions_folder = "particles"
# os.makedirs(distributions_folder, exist_ok=True)
# for idx_chunk, my_list in enumerate(particle_list):
#     pd.DataFrame(
#         my_list,
#         columns=["particle_id", "normalized amplitude in xy-plane", "angle in xy-plane [deg]"],
#     ).to_parquet(f"{distributions_folder}/{idx_chunk:02}.parquet")

# ==================================================================================================
# --- Build collider from mad model
# ==================================================================================================
config_mad_model = config_collider["config_mad"]

# Make mad environment
xm.make_mad_environment(links=config_mad_model["links"])

# Start mad
mad_b1b2 = Madx(command_log="mad_collider.log")
mad_b4 = Madx(command_log="mad_b4.log")

# Build sequences
ost.build_sequence(mad_b1b2, mylhcbeam=1)
ost.build_sequence(mad_b4, mylhcbeam=4)

# Apply optics (only for b1b2, b4 will be generated from b1b2)
ost.apply_optics(mad_b1b2, optics_file=config_mad_model["optics_file"])

# Build xsuite collider
collider = xlhc.build_xsuite_collider(
    sequence_b1=mad_b1b2.sequence.lhcb1,
    sequence_b2=mad_b1b2.sequence.lhcb2,
    sequence_b4=mad_b4.sequence.lhcb2,
    beam_config=config_mad_model["beam_config"],
    enable_imperfections=config_mad_model["enable_imperfections"],
    enable_knob_synthesis=config_mad_model["enable_knob_synthesis"],
    rename_coupling_knobs=config_mad_model["rename_coupling_knobs"],
    pars_for_imperfections=config_mad_model["pars_for_imperfections"],
    ver_lhc_run=config_mad_model["ver_lhc_run"],
    ver_hllhc_optics=config_mad_model["ver_hllhc_optics"],
)

# Remove all the temporaty files created in the process of building collider
os.remove("mad_collider.log")
os.remove("mad_b4.log")
shutil.rmtree("temp")
os.unlink("errors")
os.unlink("acc-models-lhc")

# ==================================================================================================
# ---Save to json and log result
# ==================================================================================================
os.makedirs("collider", exist_ok=True)
collider.to_json("collider/collider.json")

if tree_maker is not None:
    tree_maker.tag_json.tag_it(configuration["log_file"], "completed")
