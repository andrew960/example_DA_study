# %%
import xtrack as xt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xpart as xp
import xobjects as xo
import configure_and_track as configure_and_track
# %%
collider = xt.Multiline.from_json('/afs/cern.ch/work/a/afornara/public/run3/example_DA_study'
                                  '/master_study/master_jobs/2_configure_and_track/collider.json')

config, config_sim, config_collider = configure_and_track.read_configuration()
# %%
collider['lhcb1'].element_names = [element for element in collider['lhcb1'].element_names if not element.startswith('bb_lr')]

HO_1 = False
HO_2 = False
HO_5 = False
HO_8 = False

start_len = len(collider['lhcb1'].element_names)
if(HO_5 == False):
    print('Removing HO lenses at IP5')
    collider['lhcb1'].element_names = [element for element in collider['lhcb1'].element_names if not element.startswith('bb_ho.l5b1_')]
    collider['lhcb1'].element_names = [element for element in collider['lhcb1'].element_names if not element.startswith('bb_ho.c5b1_')]
    collider['lhcb1'].element_names = [element for element in collider['lhcb1'].element_names if not element.startswith('bb_ho.r5b1_')]
    
if(HO_8 == False):
    print('Removing HO lenses at IP8')
    collider['lhcb1'].element_names = [element for element in collider['lhcb1'].element_names if not element.startswith('bb_ho.l8b1_')]
    collider['lhcb1'].element_names = [element for element in collider['lhcb1'].element_names if not element.startswith('bb_ho.c8b1_')]
    collider['lhcb1'].element_names = [element for element in collider['lhcb1'].element_names if not element.startswith('bb_ho.r8b1_')]
if(HO_2 == False):
    print('Removing HO lenses at IP2')
    collider['lhcb1'].element_names = [element for element in collider['lhcb1'].element_names if not element.startswith('bb_ho.l2b1_')]
    collider['lhcb1'].element_names = [element for element in collider['lhcb1'].element_names if not element.startswith('bb_ho.c2b1_')]
    collider['lhcb1'].element_names = [element for element in collider['lhcb1'].element_names if not element.startswith('bb_ho.r2b1_')]
if(HO_1 == False):
    print('Removing HO lenses at IP1')
    collider['lhcb1'].element_names = [element for element in collider['lhcb1'].element_names if not element.startswith('bb_ho.l1b1_')]
    collider['lhcb1'].element_names = [element for element in collider['lhcb1'].element_names if not element.startswith('bb_ho.c1b1_')]
    collider['lhcb1'].element_names = [element for element in collider['lhcb1'].element_names if not element.startswith('bb_ho.r1b1_')]

for element in collider['lhcb1'].element_names:
    if element.startswith('bb_ho'):
        if ((element.endswith('_12')) or (element.endswith('_05')) or (element.endswith('_04')) or (element.endswith('_03')) or (element.endswith('_02')) or ((element.endswith('_01'))) ):
            collider['lhcb1'].element_names.remove(element)
        print(element)
end_len = len(collider['lhcb1'].element_names)
if(start_len != end_len):
    print(f'HO lenses removed = {start_len-end_len}')
# %%
collider.build_trackers()
# %%

def set_orbit_flat(collider):
    print('Setting optics as flat')
    for ii in ['on_x1', 'on_sep1', 'on_x2h', 'on_sep2h', 'on_x2v', 'on_sep2v', 'on_x5', 
               'on_sep5', 'on_x8h', 'on_sep8h', 'on_x8v', 'on_sep8v', 'on_disp', 
               'on_alice_normalized', 'on_lhcb_normalized','on_sol_atlas', 'on_sol_cms', 
               'on_sol_alice', 'i_oct_b1', 'i_oct_b2']:
        collider.vars[ii] = 0

def set_orbit_from_config(collider, config):
    print('Setting optics as from config')
    for ii in ['on_x1', 'on_sep1', 'on_x2h', 'on_sep2h', 'on_x2v', 'on_sep2v', 'on_x5', 
               'on_sep5', 'on_x8h', 'on_sep8h', 'on_x8v', 'on_sep8v', 'on_disp', 
               'on_alice_normalized', 'on_lhcb_normalized', 'on_sol_atlas', 'on_sol_cms', 
               'on_sol_alice', 'i_oct_b1', 'i_oct_b2']:
        collider.vars[ii] = config['config_collider']['config_knobs_and_tuning']['knob_settings'][ii]
# %%
#collider.vars['beambeam_scale'] = 0
set_orbit_flat(collider)
collider.vars['vrf400'] = 12
twiss_b1 = collider['lhcb1'].twiss()
twiss_b2 = collider['lhcb2'].twiss().reverse()

plt.plot(twiss_b1['s', : ], twiss_b1['x', : ],label='x')
plt.plot(twiss_b1['s', : ], twiss_b1['y', : ],label='y')
plt.axvline(twiss_b1[['s'],'ip8'],color = 'green', linestyle='-.', label='IP8')
plt.axvline(twiss_b1[['s'],'ip1'],color = 'black', linestyle='-.', label='IP1')
plt.axvline(twiss_b1[['s'],'ip2'],color = 'red', linestyle='-.', label='IP2')
plt.title(f'Closed orbit no HOBB lenses')
#plt.ylim(-2e-17,2e-17)
plt.legend()
plt.grid(True)
# %%
#Dispersion study
set_orbit_flat(collider)
#set_orbit_from_config(collider, config)
collider.vars['vrf400'] = 12
twiss_b1 = collider['lhcb1'].twiss()
twiss_b2 = collider['lhcb2'].twiss().reverse()
#plt.plot(twiss_b1['s', : ], twiss_b1['dx', : ],label='x')
plt.plot(twiss_b1['s', : ], twiss_b1['dy', : ],label='y')
plt.axvline(twiss_b1[['s'],'ip8'],color = 'green', linestyle='-.', label='IP8',alpha = 0.5)
plt.axvline(twiss_b1[['s'],'ip1'],color = 'black', linestyle='-.', label='IP1',alpha = 0.5)
plt.axvline(twiss_b1[['s'],'ip2'],color = 'red', linestyle='-.', label='IP2',alpha = 0.5)
plt.title(f'Dispersion no HOBB lenses')
#print all values of twiss_b1['dy', : ] that are not zero
print(twiss_b1['dy', : ][twiss_b1['dy', : ] > 1e-8])
#plt.ylim(-4e-7,4e-7)
#plt.xlim(twiss_b1[['s'],'ip8']-100,twiss_b1[['s'],'ip8']+100)
#plt.xlim(twiss_b1[['s'],'ip2']-40,twiss_b1[['s'],'ip2']+40)
#plt.xlim(twiss_b1[['s'],'ip8']-100,twiss_b1[['s'],'ip2']+100)
plt.legend()
plt.grid(True)

# %%
#find the first non zero dispersion value in twiss_b1['dy', : ]
twiss_b1['dy', : ][twiss_b1['dy', : ] > 1e-40]
#get the index of the first non zero dispersion value in twiss_b1['dy', : ]
index = np.where(np.abs(twiss_b1['dy', : ]) > 1e-40)[0][0]

print(twiss_b1['name'][index-2:index+2])
print(twiss_b1['dy'][index-2:index+2])


my_line = xt.Line(
    elements=collider['lhcb1'].element_dict,
    element_names=collider['lhcb1'].element_names)
my_line.build_tracker()
my_particle = xp.Particles(
             p0c=450e9,q0=1,mass0=xp.PROTON_MASS_EV,
             x = 0.1, px = 0.0,  y = 0.0, py = 0.0, zeta =0.0, delta = 1)
my_particle.show()
my_line.track(my_particle)
my_particle.show()




# %%
twiss_b1['dy', : ][twiss_b1['dy', : ] > 1e-40]
index = np.where(np.abs(twiss_b1['dy', : ]) > 1e-40)[0][0]
jump_index = np.where(np.abs(twiss_b1['dy', : ]) > 1e-7)[0][-1]
print(twiss_b1['name'][index-2:index+2])
print(twiss_b1['dy'][index-2:index+2])
my_index = collider['lhcb1'].element_names.index(twiss_b1['name'][index])
#plot the dispersion function until the first non zero value
#increase plot size
plt.rcParams['figure.figsize'] = [15, 5]
plt.plot(twiss_b1['s', : ], twiss_b1['dy', : ],label='dy')
plt.axvline(twiss_b1[['s'],'ip8'],color = 'green', linestyle='-.', label='IP8',alpha = 0.5)

plt.axvline(twiss_b1[['s']][jump_index],color = 'red', linestyle='-.', label=f"{twiss_b1['name'][jump_index]},{type(collider['lhcb1'].element_dict[twiss_b1['name'][jump_index]])}",alpha = 0.5)
plt.axvline(twiss_b1[['s']][my_index-1],color = 'orange', linestyle='-.', label=f"{twiss_b1['name'][my_index-1]},{type(collider['lhcb1'].element_dict[twiss_b1['name'][my_index-1]])}",alpha = 0.5)

plt.title(f'Dispersion no HOBB lenses (Everything is 0.0 until around IP8)')
plt.ylim(-4e-22,4e-22)
#plt.xlim(twiss_b1[['s'],'ip8']-100,twiss_b1[['s'],'ip8']+100)
plt.legend(fontsize=10)
plt.grid(True) 


#%%
#The problem arises with the first roation
my_line = xt.Line(
    elements=(collider['lhcb1'].element_dict),
              #|  {'my_SRotation': xt.SRotation(cos_z=0.9999101428712823, sin_z=0.01340545348475305)}),
    element_names=[ii for ii in collider['lhcb1'].element_names][0:my_index])
                #+['my_SRotation'])

my_line.particle_ref = xp.Particles(
                    mass0=xp.PROTON_MASS_EV, q0=1, energy0=0.450e12)

aux = xp.Particles(
                    mass0=xp.PROTON_MASS_EV, q0=1, energy0=0.450e12, delta=1e-4)
my_line.build_tracker()
print('---------Particle before tracking------')
aux.show()
my_line.track(aux)
print('---------Particle after tracking-------')
aux.show()

# %%