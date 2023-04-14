import numpy as np
import xobjects as xo
import xtrack as xt
import xpart as xp
import json
import pandas as pd
from cpymad.madx import Madx
import NAFFlib
import os
from math import modf
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from scipy.stats import linregress
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib import cm
import math
from scipy.stats import norm
from scipy.stats import kde
from scipy.integrate import quad
from scipy.optimize import curve_fit
import itertools
import time
from datetime import datetime
from scipy import stats
import scipy as sp
import abel
from abel.direct import direct_transform
from abel.tools.analytical import GaussianAnalytical
from utilsaf import *
import pyarrow.parquet as pq
import yaml
import logging
####################
# Read config file #
####################

with open('config.yaml','r') as fid:
    config=yaml.safe_load(fid)

######################
# Tag job as started #
######################

try:
    import tree_maker
    tree_maker.tag_json.tag_it(config['log_file'], 'started')
except ImportError:
    logging.warning('tree_maker not available')
    tree_maker = None

####################

print(f"\n\nCrab Cavities+Noise and Beam Beam study")
#Choose context, line, N of turns, emittances...
#------------------------------------------------------------
ctx = xo.ContextCupy() #Code is intended for GPU
#ctx = xo.ContextCpu()
print('Top energy optics selected')
p0c = config['p0c']
sigma_z = config['sigma_z']
normal_emitt_x = config['normal_emitt_x']
normal_emitt_y = config['normal_emitt_y']

input = config['xline_json']#'../001_machine_model/line_bb_for_tracking.json'

with open(input, 'r') as fid:
    loaded_dct = json.load(fid)
line = xt.Line.from_dict(loaded_dct)

print('Using line:'+input)
print('Line ready')

N=config['n_turns'] #Total number of turns
sampling= 1 #Sampling rate, 1 sample every 'sampling' turns
scale_noise = 1


#Tracker section
#---------------------------------------------------------------------------------
particle_0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=p0c, x=1e-5, y=1e-5)
tracker_normal = xt.Tracker(_context=ctx, line=line)
ref = tracker_normal.find_closed_orbit(particle_ref = particle_0)
tw_normal= tracker_normal.twiss(ref)
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
print('dataframe ready')
#Generating the particles


# distr = config['particle_file']
# df = pq.read_table(distr).to_pandas()
# x_in_sigmas = df['x_in_sigmas'].values
# px_in_sigmas = df['px_in_sigmas'].values
# y_in_sigmas = df['y_in_sigmas'].values
# py_in_sigmas = df['py_in_sigmas'].values

N_particles = 20000

bunch_intensity = config['bunch_intensity']
# zeta, delta = xp.generate_longitudinal_coordinates(
#     particle_ref=p0_normal,
#     num_particles=N_particles, distribution='gaussian',
#     sigma_z=7.5e-2, tracker=tracker_normal)

# particles = xp.build_particles(
#     tracker=tracker_normal,
#     particle_ref=p0_normal,
#     zeta=zeta, delta=delta,
#     x_norm=x_in_sigmas, px_norm=px_in_sigmas,
#     y_norm=y_in_sigmas, py_norm=py_in_sigmas,
#     scale_with_transverse_norm_emitt=(normal_emitt_x, normal_emitt_y))

# print('Generating N_particles =', N_particles)
# particles = xp.generate_matched_gaussian_bunch(
#                 num_particles=N_particles, total_intensity_particles=bunch_intensity,
#                 nemitt_x=normal_emitt_x, nemitt_y=normal_emitt_y, sigma_z=sigma_z,
#                 particle_ref=p0_normal,
#                 tracker=tracker_normal)

qx = 1.1
q_prime_x = (1+qx)/(3-qx)
U1x = np.random.uniform(size = N_particles)
U2x = np.random.uniform(size = N_particles)
basex = q_prime_x

Rx = np.sqrt(-2 * log_q(q_prime_x, U1x))
Thetax = 2 * np.pi * U2x
x_in_sigmas = Rx * np.cos(Thetax)
px_in_sigmas= Rx * np.sin(Thetax)

qy = 1.1
q_prime_y = (1+qy)/(3-qy)
U1y = np.random.uniform(size = N_particles)
U2y = np.random.uniform(size = N_particles)
basey = q_prime_y

Ry = Rx#np.sqrt(-2 * log_q(q_prime_y, U1y))
Thetay = 2 * np.pi * U2y
y_in_sigmas = Rx * np.cos(Thetay)
py_in_sigmas = Rx * np.sin(Thetay)




zeta, delta = xp.generate_longitudinal_coordinates(
    particle_ref=p0_normal,
    num_particles=N_particles, distribution='gaussian',
    sigma_z=7.5e-2, tracker=tracker_normal)

particles = xp.build_particles(
    tracker=tracker_normal,
    particle_ref=p0_normal,
    zeta=zeta, delta=delta,
    x_norm=x_in_sigmas, px_norm=px_in_sigmas,
    y_norm=y_in_sigmas, py_norm=py_in_sigmas,
    scale_with_transverse_norm_emitt=(normal_emitt_x, normal_emitt_y))

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
    if(ii<=100):
        xs_i.append(ctx.nparray_from_context_array(particles.x))
        pxs_i.append(ctx.nparray_from_context_array(particles.px)) 
        ys_i.append(ctx.nparray_from_context_array(particles.y))
        pys_i.append(ctx.nparray_from_context_array(particles.py))
        zetas_i.append(ctx.nparray_from_context_array(particles.zeta))
        deltas_i.append(ctx.nparray_from_context_array(particles.delta))
        states_i.append(ctx.nparray_from_context_array(particles.state))
    if(ii>=N-100):
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