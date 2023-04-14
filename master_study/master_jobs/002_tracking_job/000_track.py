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
#Creating Halo
freq = 400789598.98582596#1/8.892446333483922e-05
c = 299792458
length = tw_normal['circumference']
q0 = 1
voltage = 16e6
slip_factor = tw_normal['slip_factor']
beta0 = particle_0.beta0
A = q0*voltage/(2.*np.pi*freq*p0c/c*length)
B = 2*np.pi*freq/c
C = abs(slip_factor)/(2.*beta0*beta0)
xx = np.linspace(-np.pi/B, np.pi/B, 10000)
xx2 = np.linspace(-np.pi/B, np.pi/B, 10000)
yy = np.sqrt(2*A/C) * np.cos(B/2.*xx)
yy2 = 0.98*np.sqrt(2*A/C) * np.cos(B/2.*xx)
n_part = int(2.5e5)
sigma_z=7.5e-2
sigma_dp = 9.66128190911531e-05
(zeta_in_sigmas, delta_in_sigmas, r_points, theta_points
    )= xp.generate_2D_uniform_circular_sector(
                                        num_particles=n_part,
                                        r_range=(0, 6), # sigmas
                                        theta_range=(0, 2*np.pi))

z = []
dp = []
#plt.plot(zuni, duni, 'o', markersize=1,alpha = 1)
enclose = 0.98
for ii in range(len(delta_in_sigmas)):
    if(delta_in_sigmas[ii]*sigma_dp< enclose*np.sqrt(2*A/C) * np.cos(B/2.*zeta_in_sigmas[ii]/enclose*sigma_z)and(delta_in_sigmas[ii]*sigma_dp> -enclose*np.sqrt(2*A/C) * np.cos(B/2.*zeta_in_sigmas[ii]/enclose*sigma_z))):
        z.append(zeta_in_sigmas[ii])
        dp.append(delta_in_sigmas[ii])
n_sigma = 6.0
n_part = len(z)

x  = np.zeros(n_part)
px = np.zeros(n_part)
y  = np.zeros(n_part)
py = np.zeros(n_part)
z = np.array(z)
dp = np.array(dp)

def cmp_weights(df):
    r2 = df['x']**2 + df['px']**2 + df['y']**2 + df['py']**2
    w = np.exp(-r2/2.)
    r2_l = df['z']**2 + df['dp']**2
    w *=np.exp(-r2_l/2.)
    w/=np.sum(w)
    return w

def generate_pseudoKV_xpyp(i):
  not_generated = True
  while not_generated:
    u = np.random.normal(size=4)
    r = np.sqrt(np.sum(u**2))
    u *= n_sigma/r
    v = np.random.normal(size=4)
    r = np.sqrt(np.sum(v**2))
    v *= n_sigma/r
    R2 = u[0]**2 + u[1]**2 + v[0]**2 + v[1]**2 
    if R2 <= n_sigma**2:
        x[i]  = u[0]
        px[i] = u[1]
        y[i]  = v[0]
        py[i] = v[1]
        #z[i] = u[0]
        #dp[i] = v[1]
        not_generated = False
  return 
list(map(generate_pseudoKV_xpyp, range(len(x))))

df = pd.DataFrame({'x':x , 'y': y, 'px': px, 'py': py, 'z': z, 'dp':dp})
df['weights'] = cmp_weights(df)

x_in_sigmas = np.array(df['x'])
y_in_sigmas = np.array(df['y'])
px_in_sigmas = np.array(df['px'])
py_in_sigmas = np.array(df['py'])
z_in_sigmas = np.array(df['z'])
dp_in_sigmas = np.array(df['dp'])

particles = xp.build_particles(
    tracker=tracker_normal,
    particle_ref=p0_normal,
    zeta=z*sigma_z, delta=dp*sigma_dp,
    x_norm=x_in_sigmas, px_norm=px_in_sigmas,
    y_norm=y_in_sigmas, py_norm=py_in_sigmas,
    scale_with_transverse_norm_emitt=(normal_emitt_x, normal_emitt_y))

N_particles = len(z_in_sigmas)


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
np.savez('halostudy_parameters.npz', N_particles=N_particles, N=N, a_noise=a_noise, ph_noise=ph_noise, sampling = 33333)
start = time.time()
print("Begin tracking N_particles = ", N_particles)
print("Total turns =", N)
print('------------------------------------')
aa = 0
for ii in range(N):
    if(ii%33333 == 0):
        xs_i.append(ctx.nparray_from_context_array(particles.x))
        pxs_i.append(ctx.nparray_from_context_array(particles.px)) 
        ys_i.append(ctx.nparray_from_context_array(particles.y))
        pys_i.append(ctx.nparray_from_context_array(particles.py))
        zetas_i.append(ctx.nparray_from_context_array(particles.zeta))
        deltas_i.append(ctx.nparray_from_context_array(particles.delta))
        states_i.append(ctx.nparray_from_context_array(particles.state))
        gemittx = normal_emitt_x/particle_0.gamma0/particle_0.beta0
        scale = 1/(np.sqrt(gemittx))
        X_i = get_normalized_phase_space(ctx.nparray_from_context_array(particles.x)-ctx.nparray_from_context_array(particles.delta)*dx_3,ctx.nparray_from_context_array(particles.px)-ctx.nparray_from_context_array(particles.delta)*dpx_3,betx_3,alphx_3,scale)
        Y_i = get_normalized_phase_space(ctx.nparray_from_context_array(particles.y)-ctx.nparray_from_context_array(particles.delta)*dy_3,ctx.nparray_from_context_array(particles.py)-ctx.nparray_from_context_array(particles.delta)*dpy_3,bety_3,alphy_3,scale)
        Z_i = ctx.nparray_from_context_array(particles.zeta)/sigma_z
        DP_i = ctx.nparray_from_context_array(particles.delta)/sigma_dp
        df_normalized = pd.DataFrame({'x':X_i[0].flatten(),'px':X_i[1].flatten(),'y':Y_i[0].flatten(),'py':Y_i[1].flatten(),'z':Z_i.flatten(),'dp':DP_i.flatten(),'states':ctx.nparray_from_context_array(particles.state)})
        df_normalized['weights'] = cmp_weights(df)
        df_normalized.to_parquet(f'halostudy_normalized_{aa}.parquet')
        aa+=1
        end = time.time()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print('Round = ',ii,'Time(s) = ',end-start,'Current Time =', current_time)
        currstates = ctx.nparray_from_context_array(particles.state)
        print('Lost particles = ', len(currstates[currstates == -1]))
        if(True):
            set_crabs_IP5(line,'acfgav.4bl5.b1',0,ii,a_noise,ph_noise,mu_knl,mu_ksl,mu_pn,mu_ps)
            set_crabs_IP5(line,'acfgav.4al5.b1',2,ii,a_noise,ph_noise,mu_knl,mu_ksl,mu_pn,mu_ps)
            set_crabs_IP5(line,'acfgav.4ar5.b1',4,ii,a_noise,ph_noise,mu_knl,mu_ksl,mu_pn,mu_ps)
            set_crabs_IP5(line,'acfgav.4br5.b1',6,ii,a_noise,ph_noise,mu_knl,mu_ksl,mu_pn,mu_ps)

            set_crabs_IP1(line,'acfgah.4bl1.b1',9,ii,a_noise,ph_noise,mu_knl,mu_ksl,mu_pn,mu_ps)
            set_crabs_IP1(line,'acfgah.4al1.b1',11,ii,a_noise,ph_noise,mu_knl,mu_ksl,mu_pn,mu_ps)
            set_crabs_IP1(line,'acfgah.4ar1.b1',13,ii,a_noise,ph_noise,mu_knl,mu_ksl,mu_pn,mu_ps)
            set_crabs_IP1(line,'acfgah.4br1.b1',15,ii,a_noise,ph_noise,mu_knl,mu_ksl,mu_pn,mu_ps)

    tracker_normal.track(particles, num_turns=sampling,turn_by_turn_monitor=False)



xs_i = np.array(xs_i)
pxs_i = np.array(pxs_i)
ys_i = np.array(ys_i)
pys_i = np.array(pys_i)
zetas_i = np.array(zetas_i)
deltas_i = np.array(deltas_i)
states_i = np.array(states_i)


#save xs_i, pxs_i, ys_i, pys_i, zetas_i, deltas_i, states_i in parquet file
df_physical = pd.DataFrame({'x':xs_i.flatten(),'px':pxs_i.flatten(),'y':ys_i.flatten(),'py':pys_i.flatten(),'z':zetas_i.flatten(),'dp':deltas_i.flatten(),'states':states_i.flatten()})
df_physical.to_parquet('halostudy_physical_.parquet')

if tree_maker is not None:
    tree_maker.tag_json.tag_it(config['log_file'], 'completed')