# ==================================================================================================
# --- Imports
# ==================================================================================================
import json
import yaml
import time
import logging
import numpy as np
import pandas as pd
import os
import xtrack as xt
import tree_maker
import xmask as xm
import xmask.lhc as xlhc
from gen_config_orbit_correction import generate_orbit_correction_setup

# ==================================================================================================
# --- Read configuration files and tag start of the job
# ==================================================================================================
# Read configuration for simulations
with open("config.yaml", "r") as fid:
    config = yaml.safe_load(fid)
config_sim = config["config_simulation"]
config_collider = config["config_collider"]

# Start tree_maker logging if log_file is present in config
if tree_maker is not None and "log_file" in config:
    tree_maker.tag_json.tag_it(config["log_file"], "started")
else:
    logging.warning("tree_maker loging not available")


# ==================================================================================================
# --- Rebuild collider
# ==================================================================================================
# Load collider and build trackers
collider = xt.Multiline.from_json(config_sim["collider_file"])


# ==================================================================================================
# --- Generate config correction files
# ==================================================================================================
correction_setup = generate_orbit_correction_setup()
os.makedirs("correction", exist_ok=True)
for nn in ["lhcb1", "lhcb2"]:
    with open(f"correction/corr_co_{nn}.json", "w") as fid:
        json.dump(correction_setup[nn], fid, indent=4)

# ==================================================================================================
# --- Install beam-beam
# ==================================================================================================
config_bb = config_collider["config_beambeam"]

# Install beam-beam lenses (inactive and not configured)
collider.install_beambeam_interactions(
    clockwise_line="lhcb1",
    anticlockwise_line="lhcb2",
    ip_names=["ip1", "ip2", "ip5", "ip8"],
    delay_at_ips_slots=[0, 891, 0, 2670],
    num_long_range_encounters_per_side=config_bb["num_long_range_encounters_per_side"],
    num_slices_head_on=config_bb["num_slices_head_on"],
    harmonic_number=35640,
    bunch_spacing_buckets=config_bb["bunch_spacing_buckets"],
    sigmaz=config_bb["sigma_z"],
)

# ==================================================================================================
# ---Knobs and tuning
# ==================================================================================================
# Build trackers
collider.build_trackers()

# Read knobs and tuning settings from config file
conf_knobs_and_tuning = config_collider["config_knobs_and_tuning"]

# Set all knobs (crossing angles, dispersion correction, rf, crab cavities,
# experimental magnets, etc.)
for kk, vv in conf_knobs_and_tuning["knob_settings"].items():
    collider.vars[kk] = vv

# Tunings
for line_name in ["lhcb1", "lhcb2"]:
    knob_names = conf_knobs_and_tuning["knob_names"][line_name]

    targets = {
        "qx": conf_knobs_and_tuning["qx"][line_name],
        "qy": conf_knobs_and_tuning["qy"][line_name],
        "dqx": conf_knobs_and_tuning["dqx"][line_name],
        "dqy": conf_knobs_and_tuning["dqy"][line_name],
    }

    xm.machine_tuning(
        line=collider[line_name],
        enable_closed_orbit_correction=True,
        enable_linear_coupling_correction=True,
        enable_tune_correction=True,
        enable_chromaticity_correction=True,
        knob_names=knob_names,
        targets=targets,
        line_co_ref=collider[line_name + "_co_ref"],
        co_corr_config=conf_knobs_and_tuning["closed_orbit_correction"][line_name],
    )

# ==================================================================================================
# --- Compute the number of collisions in the different IPs (used for luminosity leveling)
# ==================================================================================================

# Get the filling scheme path (in json or csv format)
filling_scheme_path = config_bb["mask_with_filling_pattern"]["pattern_fname"]

# Load the filling scheme
if filling_scheme_path.endswith(".json"):
    with open(filling_scheme_path, "r") as fid:
        filling_scheme = json.load(fid)
else:
    raise ValueError(
        f"Unknown filling scheme file format: {filling_scheme_path}. It you provided a csv file, it"
        " should have been automatically convert when running the script 001_make_folders.py."
        " Something went wrong."
    )

# Extract booleans beam arrays
array_b1 = np.array(filling_scheme["beam1"])
array_b2 = np.array(filling_scheme["beam2"])

# Assert that the arrays have the required length, and do the convolution
assert len(array_b1) == len(array_b2) == 3564
n_collisions_ip1_and_5 = array_b1 @ array_b2
n_collisions_ip2 = np.roll(array_b1, -891) @ array_b2
n_collisions_ip8 = np.roll(array_b1, -2670) @ array_b2

# ==================================================================================================
# ---Levelling
# ==================================================================================================
if "config_lumi_leveling" in config_collider and not config_collider["skip_leveling"]:
    # Read knobs and tuning settings from config file (already updated with the number of collisions)
    config_lumi_leveling = config_collider["config_lumi_leveling"]

    # Update the number of bunches in the configuration file
    config_lumi_leveling["ip8"]["num_colliding_bunches"] = int(n_collisions_ip8)

    # Level luminosity
    xlhc.luminosity_leveling(
        collider, config_lumi_leveling=config_lumi_leveling, config_beambeam=config_bb
    )

    # Re-match tunes, and chromaticities
    for line_name in ["lhcb1", "lhcb2"]:
        knob_names = conf_knobs_and_tuning["knob_names"][line_name]
        targets = {
            "qx": conf_knobs_and_tuning["qx"][line_name],
            "qy": conf_knobs_and_tuning["qy"][line_name],
            "dqx": conf_knobs_and_tuning["dqx"][line_name],
            "dqy": conf_knobs_and_tuning["dqy"][line_name],
        }
        xm.machine_tuning(
            line=collider[line_name],
            enable_tune_correction=True,
            enable_chromaticity_correction=True,
            knob_names=knob_names,
            targets=targets,
        )

else:
    print(
        "No leveling is done as no configuration has been provided, or skip_leveling"
        " is set to True."
    )

# ==================================================================================================
# --- Add linear coupling and rematch tune and chromaticity
# ==================================================================================================

# Add linear coupling as the target in the tuning of the base collider was 0
# (not possible to set it the target to 0.001 for now)
# ! This is commented as this affects the tune/chroma too much
# ! We need to wait for the possibility to set the linear coupling as a target along with tune/chroma
# collider.vars["c_minus_re_b1"] += conf_knobs_and_tuning["delta_cmr"]
# collider.vars["c_minus_re_b2"] += conf_knobs_and_tuning["delta_cmr"]

# Rematch tune and chromaticity
for line_name in ["lhcb1", "lhcb2"]:
    knob_names = conf_knobs_and_tuning["knob_names"][line_name]
    targets = {
        "qx": conf_knobs_and_tuning["qx"][line_name],
        "qy": conf_knobs_and_tuning["qy"][line_name],
        "dqx": conf_knobs_and_tuning["dqx"][line_name],
        "dqy": conf_knobs_and_tuning["dqy"][line_name],
    }
    xm.machine_tuning(
        line=collider[line_name],
        enable_tune_correction=True,
        enable_chromaticity_correction=True,
        enable_linear_coupling_correction=False,
        knob_names=knob_names,
        targets=targets,
    )

# ==================================================================================================
# --- Assert that tune, chromaticity and linear coupling are correct before going further
# ==================================================================================================
for line_name in ["lhcb1", "lhcb2"]:
    tw = collider[line_name].twiss()
    assert np.isclose(tw.qx, conf_knobs_and_tuning["qx"][line_name], atol=1e-4), (
        f"tune_x is not correct for {line_name}. Expected {conf_knobs_and_tuning['qx'][line_name]},"
        f" got {tw.qx}"
    )
    assert np.isclose(tw.qy, conf_knobs_and_tuning["qy"][line_name], atol=1e-4), (
        f"tune_y is not correct for {line_name}. Expected {conf_knobs_and_tuning['qy'][line_name]},"
        f" got {tw.qy}"
    )
    assert np.isclose(
        tw.dqx,
        conf_knobs_and_tuning["dqx"][line_name],
        rtol=1e-2,
    ), (
        f"chromaticity_x is not correct for {line_name}. Expected"
        f" {conf_knobs_and_tuning['dqx'][line_name]}, got {tw.dqx}"
    )
    assert np.isclose(
        tw.dqy,
        conf_knobs_and_tuning["dqy"][line_name],
        rtol=1e-2,
    ), (
        f"chromaticity_y is not correct for {line_name}. Expected"
        f" {conf_knobs_and_tuning['dqy'][line_name]}, got {tw.dqy}"
    )
    # ! Commented as the linear coupling is not optimized anymore
    # ! This should be updated when possible
    # assert np.isclose(
    #     tw.c_minus,
    #     conf_knobs_and_tuning["delta_cmr"],
    #     atol=5e-3,
    # ), (
    #     f"linear coupling is not correct for {line_name}. Expected"
    #     f" {conf_knobs_and_tuning['delta_cmr']}, got {tw.c_minus}"
    # )

# ==================================================================================================
# --- Configure beam-beam
# ==================================================================================================
print("Configuring beam-beam lenses...")
collider.configure_beambeam_interactions(
    num_particles=config_bb["num_particles_per_bunch"],
    nemitt_x=config_bb["nemitt_x"],
    nemitt_y=config_bb["nemitt_y"],
)

# Configure filling scheme mask and bunch numbers
if "mask_with_filling_pattern" in config_bb:
    # Initialize filling pattern with empty values
    filling_pattern_cw = None
    filling_pattern_acw = None

    # Initialize bunch numbers with empty values
    i_bunch_cw = None
    i_bunch_acw = None

    if "pattern_fname" in config_bb["mask_with_filling_pattern"]:
        # Fill values if possible
        if config_bb["mask_with_filling_pattern"]["pattern_fname"] is not None:
            fname = config_bb["mask_with_filling_pattern"]["pattern_fname"]
            with open(fname, "r") as fid:
                filling = json.load(fid)
            filling_pattern_cw = filling["beam1"]
            filling_pattern_acw = filling["beam2"]

            # Only track bunch number if a filling pattern has been provided
            if "i_bunch_b1" in config_bb["mask_with_filling_pattern"]:
                i_bunch_cw = config_bb["mask_with_filling_pattern"]["i_bunch_b1"]
            if "i_bunch_b2" in config_bb["mask_with_filling_pattern"]:
                i_bunch_acw = config_bb["mask_with_filling_pattern"]["i_bunch_b2"]

            # Note that a bunch number must be provided if a filling pattern is provided
            # Apply filling pattern
            collider.apply_filling_pattern(
                filling_pattern_cw=filling["beam1"],
                filling_pattern_acw=filling["beam2"],
                i_bunch_cw=i_bunch_cw,
                i_bunch_acw=i_bunch_acw,
            )

# ==================================================================================================
# --- Save the final collider before tracking
# ==================================================================================================
collider.to_json("final_collider.json")

# ==================================================================================================
# --- Prepare particles distribution for tracking
# ==================================================================================================
particle_df = pd.read_parquet(config_sim["particle_file"])

r_vect = particle_df["normalized amplitude in xy-plane"].values
theta_vect = particle_df["angle in xy-plane [deg]"].values * np.pi / 180  # [rad]

A1_in_sigma = r_vect * np.cos(theta_vect)
A2_in_sigma = r_vect * np.sin(theta_vect)

particles = collider.lhcb1.build_particles(
    x_norm=A1_in_sigma,
    y_norm=A2_in_sigma,
    delta=config_sim["delta_max"],
    scale_with_transverse_norm_emitt=(config_bb["nemitt_x"], config_bb["nemitt_y"]),
)
particles.particle_id = particle_df.particle_id.values


# ==================================================================================================
# --- Build tracker and track
# ==================================================================================================
# Optimize line for tracking
collider.lhcb1.optimize_for_tracking()

# Save initial coordinates
pd.DataFrame(particles.to_dict()).to_parquet("input_particles.parquet")

# Track
num_turns = config_sim["n_turns"]
a = time.time()
collider.lhcb1.track(particles, turn_by_turn_monitor=False, num_turns=num_turns)
b = time.time()

print(f"Elapsed time: {b-a} s")
print(f"Elapsed time per particle per turn: {(b-a)/particles._capacity/num_turns*1e6} us")

# ==================================================================================================
# --- Save output
# ==================================================================================================
pd.DataFrame(particles.to_dict()).to_parquet("output_particles.parquet")

if tree_maker is not None and "log_file" in config:
    tree_maker.tag_json.tag_it(config["log_file"], "completed")
