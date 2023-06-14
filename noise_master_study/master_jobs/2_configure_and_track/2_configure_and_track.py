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
import NAFFlib
import os
import math
from math import modf
from utilsaf import *
import pyarrow.parquet as pq
import yaml
import logging

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
# --- Tracking section, ctx is GPU
# ==================================================================================================
ctx = xo.ContextCupy()
p0c = config['p0c']
sigma_z = config['sigma_z']
normal_emitt_x = config['nemitt_x']
normal_emitt_y = config['nemitt_y']
input = config['xline_json']

with open(input, 'r') as fid:
    loaded_dct = json.load(fid)
line = xt.Line.from_dict(loaded_dct)

N=config['n_turns'] #Total number of turns
sampling= 1 #Sampling rate, 1 sample every 'sampling' turns
scale_noise = 1 #Scale noise by this factor

particle_0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=p0c, x=1e-5, y=1e-5)
tracker_normal = xt.Tracker(_context=ctx, line=line)
ref = line.find_closed_orbit(particle_ref = particle_0)
tw_normal= line.twiss(ref)
tw_normal['name'] = list(tw_normal['name'])
tw_df = produce_twiss_df(tw_normal)
dx_3 = tw_df[tw_df['name'] == 'ip3'].dx.values[0]
dpx_3 = tw_df[tw_df['name'] == 'ip3'].dpx.values[0]
dy_3 = tw_df[tw_df['name'] == 'ip3'].dy.values[0]
dpy_3 = tw_df[tw_df['name'] == 'ip3'].dpy.values[0]
betx_3 = tw_df[tw_df['name'] == 'ip3'].betx.values[0]
bety_3 = tw_df[tw_df['name'] == 'ip3'].bety.values[0]
alphx_3 = tw_df[tw_df['name'] == 'ip3'].alfx.values[0]
alphy_3 = tw_df[tw_df['name'] == 'ip3'].alfy.values[0]
sigma_x = np.sqrt(betx_3*normal_emitt_x/(particle_0.gamma0*particle_0.beta0))
sigma_y = np.sqrt(bety_3*normal_emitt_y/(particle_0.gamma0*particle_0.beta0))
#Closed orbit particle
p0_normal = ref 

#Studying closed orbit GPU
p0_df = pd.DataFrame({
                    'x':ctx.nparray_from_context_array(p0_normal.x),
                    'px':ctx.nparray_from_context_array(p0_normal.px),
                    'y':ctx.nparray_from_context_array(p0_normal.y),
                    'py':ctx.nparray_from_context_array(p0_normal.py),
                    'zeta':ctx.nparray_from_context_array(p0_normal.zeta),
                    'delta':ctx.nparray_from_context_array(p0_normal.delta),
                    })

N_particles = 20000
bunch_intensity = config['bunch_intensity']
print('Generating N_particles =', N_particles)
particles = xp.generate_matched_gaussian_bunch(
                num_particles=N_particles, total_intensity_particles=bunch_intensity,
                nemitt_x=normal_emitt_x, nemitt_y=normal_emitt_y, sigma_z=sigma_z,
                particle_ref=p0_normal,
                tracker=tracker_normal)


print('Generation complete')
xs_i = []
pxs_i = []
ys_i = []
pys_i = []
zetas_i = []
deltas_i = []
states_i = []

xs_f = []
pxs_f = []
ys_f = []
pys_f = []
zetas_f = []
deltas_f = []
states_f = []

emits = []

emits_x=np.zeros(N)
emits_y=np.zeros(N)

mu_knl = np.zeros(16)
mu_ksl = np.zeros(16)
mu_pn = np.zeros(16)
mu_ps = np.zeros(16)
jj = 0
for elem in line.element_dict :
    if isinstance(line.element_dict[elem],xt.beam_elements.elements.RFMultipole): 
        mu_knl[jj]=line.element_dict[elem].knl[0]
        mu_ksl[jj]=line.element_dict[elem].ksl[0]
        mu_pn[jj]=line.element_dict[elem].pn
        mu_ps[jj]=line.element_dict[elem].ps
        jj+=1

print('Loading noise files')
with open(config['noise_file'], 'rb') as f:
    ph_noise = np.load(f)
    a_noise =np.load(f)
radtodeg = 180/np.pi


print(f'Phase noise mu = {np.mean(ph_noise[0]/radtodeg)}, var = {np.var(ph_noise[0]/radtodeg)}')
print(f'Amplitude noise mu = {np.mean(a_noise[0])}, var = {np.var(a_noise[0])}')
jj = 0
start = time.time()
print("Begin tracking N_particles = ", N_particles)
print("Total turns =", N)
print('------------------------------------')
for ii in range(N):
    if(ii<=10):
        xs_i.append(ctx.nparray_from_context_array(particles.x))
        pxs_i.append(ctx.nparray_from_context_array(particles.px)) 
        ys_i.append(ctx.nparray_from_context_array(particles.y))
        pys_i.append(ctx.nparray_from_context_array(particles.py))
        zetas_i.append(ctx.nparray_from_context_array(particles.zeta))
        deltas_i.append(ctx.nparray_from_context_array(particles.delta))
        states_i.append(ctx.nparray_from_context_array(particles.state))
    if(ii>=N-10):
        xs_f.append(ctx.nparray_from_context_array(particles.x))
        pxs_f.append(ctx.nparray_from_context_array(particles.px)) 
        ys_f.append(ctx.nparray_from_context_array(particles.y))
        pys_f.append(ctx.nparray_from_context_array(particles.py))
        zetas_f.append(ctx.nparray_from_context_array(particles.zeta))
        deltas_f.append(ctx.nparray_from_context_array(particles.delta))
        states_f.append(ctx.nparray_from_context_array(particles.state))
    if(True):
        set_crabs_IP5(line,'acfgav.4bl5.b1',0,ii,a_noise,ph_noise,mu_knl,mu_ksl,mu_pn,mu_ps)
        set_crabs_IP5(line,'acfgav.4al5.b1',2,ii,a_noise,ph_noise,mu_knl,mu_ksl,mu_pn,mu_ps)
        set_crabs_IP5(line,'acfgav.4ar5.b1',4,ii,a_noise,ph_noise,mu_knl,mu_ksl,mu_pn,mu_ps)
        set_crabs_IP5(line,'acfgav.4br5.b1',6,ii,a_noise,ph_noise,mu_knl,mu_ksl,mu_pn,mu_ps)

        set_crabs_IP1(line,'acfgah.4bl1.b1',9,ii,a_noise,ph_noise,mu_knl,mu_ksl,mu_pn,mu_ps)
        set_crabs_IP1(line,'acfgah.4al1.b1',11,ii,a_noise,ph_noise,mu_knl,mu_ksl,mu_pn,mu_ps)
        set_crabs_IP1(line,'acfgah.4ar1.b1',13,ii,a_noise,ph_noise,mu_knl,mu_ksl,mu_pn,mu_ps)
        set_crabs_IP1(line,'acfgah.4br1.b1',15,ii,a_noise,ph_noise,mu_knl,mu_ksl,mu_pn,mu_ps)

    if(ii%1==0):
        #calculating emittance        
        xs_ = ctx.nparray_from_context_array(particles.x)
        pxs_ = ctx.nparray_from_context_array(particles.px)
        ys_ = ctx.nparray_from_context_array(particles.y)
        pys_ = ctx.nparray_from_context_array(particles.py)
        delta_ = ctx.nparray_from_context_array(particles.delta)
        states_ = ctx.nparray_from_context_array(particles.state)
        p0_ = ctx.nparray_from_context_array(particles.p0c[0])

        xs_1 = np.array(xs_)
        pxs_1 = np.array(pxs_)
        ys_1= np.array(ys_)
        pys_1 = np.array(pys_)
        delta_1 = np.array(delta_)
        states_1 = np.array(states_)

        cut_x = np.abs(xs_1)<6*sigma_x
        cut_y = np.abs(ys_1)<6*sigma_y

        ex = emittance(xs_1[cut_x],pxs_1[cut_x],delta_1[cut_x],dx_3,dpx_3)*particle_0.gamma0*particle_0.beta0
        ey = emittance(ys_1[cut_y],pys_1[cut_y],delta_1[cut_y],dy_3,dpy_3)*particle_0.gamma0*particle_0.beta0
        emits_x[jj]=ex
        emits_y[jj]=ey
        jj+=1

    if(ii%111111 == 0):
        end = time.time()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print('Round = ',ii,'Time(s) = ',end-start,'Current Time =', current_time)
        currstates = ctx.nparray_from_context_array(particles.state)
        print('Emittance x = ',ex,'Emittance y = ',ey)
        print('Lost particles = ', len(currstates[currstates == -1]))
    tracker_normal.track(particles, num_turns=sampling,turn_by_turn_monitor=False)

xs_i = np.array(xs_i)
pxs_i = np.array(pxs_i)
ys_i = np.array(ys_i)
pys_i = np.array(pys_i)
zetas_i = np.array(zetas_i)
deltas_i = np.array(deltas_i)
states_i = np.array(states_i)

xs_f = np.array(xs_f)
pxs_f = np.array(pxs_f)
ys_f = np.array(ys_f)
pys_f = np.array(pys_f)
zetas_f = np.array(zetas_f)
deltas_f = np.array(deltas_f)
states_f = np.array(states_f)

#save xs_i,ys_i to a parquet file
df_i = pd.DataFrame({'x':xs_i.flatten(),'px':pxs_i.flatten(),'y':ys_i.flatten(),'py':pys_i.flatten(),'zeta':zetas_i.flatten(),'delta':deltas_i.flatten(),'state':states_i.flatten()})
print('initial =',df_i)
df_i.to_parquet('initial_state.parquet')
#save xs_f,ys_f to a parquet file
df_f = pd.DataFrame({'x':xs_f.flatten(),'px':pxs_f.flatten(),'y':ys_f.flatten(),'py':pys_f.flatten(),'zeta':zetas_f.flatten(),'delta':deltas_f.flatten(),'state':states_f.flatten()})
print('final =',df_f)
df_f.to_parquet('ending_state.parquet')

#save emits_x,emits_y, ph_noise,a_noise to a parquet file
df_e = pd.DataFrame({'ex':emits_x.flatten(),'ey':emits_y.flatten()})
df_e.to_parquet('others.parquet')

if tree_maker is not None:
    tree_maker.tag_json.tag_it(config['log_file'], 'completed')























# particle_df = pd.read_parquet(config_sim["particle_file"])

# r_vect = particle_df["normalized amplitude in xy-plane"].values
# theta_vect = particle_df["angle in xy-plane [deg]"].values * np.pi / 180  # [rad]

# A1_in_sigma = r_vect * np.cos(theta_vect)
# A2_in_sigma = r_vect * np.sin(theta_vect)

# particles = collider.lhcb1.build_particles(
#     x_norm=A1_in_sigma,
#     y_norm=A2_in_sigma,
#     delta=config_sim["delta_max"],
#     scale_with_transverse_norm_emitt=(config_bb["nemitt_x"], config_bb["nemitt_y"]),
# )
# particles.particle_id = particle_df.particle_id.values


# # ==================================================================================================
# # --- Build tracker and track
# # ==================================================================================================
# # Optimize line for tracking
# collider.lhcb1.optimize_for_tracking()

# # Save initial coordinates
# pd.DataFrame(particles.to_dict()).to_parquet("input_particles.parquet")

# # Track
# num_turns = config_sim["n_turns"]
# a = time.time()
# collider.lhcb1.track(particles, turn_by_turn_monitor=False, num_turns=num_turns)
# b = time.time()

# print(f"Elapsed time: {b-a} s")
# print(f"Elapsed time per particle per turn: {(b-a)/particles._capacity/num_turns*1e6} us")

# ==================================================================================================
# --- Save output
# ==================================================================================================
# pd.DataFrame(particles.to_dict()).to_parquet("output_particles.parquet")

if tree_maker is not None and "log_file" in config:
    tree_maker.tag_json.tag_it(config["log_file"], "completed")
