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
collider.vars['on_lhcb_normalized'] = 0
collider.vars['on_alice_normalized'] = 0
collider.vars['on_sol_alice'] = 0
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
context = xo.ContextCpu()
my_list = []
names = collider['lhcb1'].element_names[collider['lhcb1'].element_names.index('ip2')-100:collider['lhcb1'].element_names.index('ip2')+100]
for ii in names:
        my_particle = xp.Particles(
             p0c=450e9,q0=1,mass0=xp.PROTON_MASS_EV,
             x = 0.1, px = 0.0,  y = 0.0, py = 0.0, zeta =0.0, delta = 1, 
             _context=context)
        collider['lhcb1'].element_dict[ii].track(my_particle)
        if(my_particle.y[0]-twiss_b1['y',ii] != 0):
            #collider['lhcb1'].element_dict[ii].sin_z = 0
            my_list.append({'name':ii,'x_part':my_particle.x[0],'y_part':my_particle.y[0],
                            'tw_x':twiss_b1['x',ii],'tw_y':twiss_b1['y',ii],'diff_x':my_particle.x[0]-twiss_b1['x',ii], 
                            'diff_y':my_particle.y[0]-twiss_b1['y',ii],'delta': my_particle.delta[0] ,'sin_z':collider['lhcb1'].element_dict[ii].sin_z})
# names = collider['lhcb1'].element_names[collider['lhcb1'].element_names.index('ip8')-100:collider['lhcb1'].element_names.index('ip8')+100]
# for ii in names:
#     ##if 'bb' in ii:
#         my_particle = xp.Particles(p0c=450e9, #eV
#              q0=1,
#              mass0=xp.PROTON_MASS_EV,x = 1e-5,  y = 0)
#         collider['lhcb1'].element_dict[ii].track(my_particle)
#         my_list.append({'name':ii,'x_part':my_particle.x[0],'y_part':my_particle.y[0],'tw_x':twiss_b1['x',ii],'tw_y':twiss_b1['y',ii],'diff_x':my_particle.x[0]-twiss_b1['x',ii], 'diff_y':my_particle.y[0]-twiss_b1['y',ii]})

        #my_list.append({'name':ii,'delta_px':my_particle.px[0],'delta_py':my_particle.py[0]})
    #print(f'element {ii}: delta_px = {my_particle.px[0]},  delta_py = {my_particle.py[0]}')
my_df = pd.DataFrame(my_list)
my_df[my_df['diff_x'] != 0]
my_df[my_df['diff_y'] != 0]

# %%
my_particle = xp.Particles(
            p0c=450e9,q0=1,mass0=xp.PROTON_MASS_EV,
            x = 0.0, px = 0.0,  y = 0.0, py = 0.0, zeta =0.0, delta = 1.0, 
            _context=context)
#mq.7r3.b1..1
#mbwmd.1l2_tilt_entry
#mb.a8r3.b1..1
#mqtli.a9r3.b1..1
my_particle.show()
collider['lhcb1'].element_dict['mbwmd.1l2_tilt_entry'].track(my_particle)
my_particle.show()

# %%
#count all the elements that start with mb in collider['lhcb1'].element_dict[ii]
dipoles = 0
for ii in collider['lhcb1'].element_names:
    if(ii.startswith('mq')):
        print(ii)

# %%






























# %% It seems that the HO is kicking...
# It takes a while to run (2-3 minutes)
import xpart as xp
my_list = []
for ii in collider['lhcb1'].element_dict:
    ##if 'bb' in ii:
        my_particle = xp.Particles(p0c=450e9, #eV
             q0=1,
             mass0=xp.PROTON_MASS_EV)
        collider['lhcb1'].element_dict[ii].track(my_particle)
        my_list.append({'name':ii,'delta_px':my_particle.px[0],'delta_py':my_particle.py[0]})
        #print(f'element {ii}: delta_px = {my_particle.px[0]},  delta_py = {my_particle.py[0]}')
my_df = pd.DataFrame(my_list)
my_df[my_df['delta_px'] != 0]
my_df[my_df['delta_py'] != 0]


#hobb_ip1 = len([element for element in collider['lhcb1'].element_names if element.startswith('bb_ho.l1b1_')])+len([element for element in collider['lhcb1'].element_names if element.startswith('bb_ho.c1b1_')])+len([element for element in collider['lhcb1'].element_names if element.startswith('bb_ho.r1b1_')])
# print(hobb_ip1)




# %%
collider['lhcb1'].element_dict['bb_lr.r1b1_01'].to_dict()
# %%
collider['lhcb1'].element_dict['bb_ho.c1b1_00'].to_dict()

# %%

# %%
# %%
#Second problem: y dispersion
collider.vars['beambeam_scale'] = 0
collider.vars['vrf400'] = 12
set_orbit_flat(collider)
twiss_b1 = collider['lhcb1'].twiss()
twiss_b2 = collider['lhcb2'].twiss().reverse()

# %%
#plt.plot(twiss_b1['s', : ], twiss_b1['dx', : ],label='x')
plt.plot(twiss_b2['s', : ], twiss_b2['dy', : ],label='y')
plt.axvline(twiss_b2[['s'],'ip8'],color = 'green', linestyle='-.', label='IP8',alpha = 0.5)
plt.axvline(twiss_b2[['s'],'ip1'],color = 'black', linestyle='-.', label='IP1',alpha = 0.5)
plt.axvline(twiss_b2[['s'],'ip2'],color = 'red', linestyle='-.', label='IP2',alpha = 0.5)
plt.title(f'Dispersion')
plt.legend()
plt.grid(True)
# %%
#We can study the rotated lattice
collider_start = xt.Multiline.from_json('/afs/cern.ch/work/a/afornara/public/run3/example_DA_study/master_study/master_jobs/1_build_distr_and_collider/collider/collider.json')

# %%
collider_start.build_trackers()

# %%
#Here the orbit is zero!
set_orbit_flat(collider_start)
# collider.vars['beambeam_scale'] = 0 
twiss_b1_start = collider_start['lhcb1'].twiss()
twiss_b2_start = collider_start['lhcb2'].twiss().reverse()

plt.plot(twiss_b1_start['s', : ], twiss_b1_start['x', : ],label='x')
plt.plot(twiss_b2_start['s', : ], twiss_b2_start['y', : ],label='y')
plt.axvline(twiss_b2_start[['s'],'ip8'],color = 'green', linestyle='-.', label='IP8')
plt.axvline(twiss_b2_start[['s'],'ip1'],color = 'black', linestyle='-.', label='IP1')
plt.axvline(twiss_b2_start[['s'],'ip2'],color = 'red', linestyle='-.', label='IP2')
plt.title(f'Closed orbit')
plt.ylim(-2e-17,2e-17)
plt.legend()
plt.grid(True)
# %%

set_orbit_flat(collider_start)
# collider.vars['beambeam_scale'] = 0 
twiss_b1_start = collider_start['lhcb1'].twiss()
twiss_b2_start = collider_start['lhcb2'].twiss().reverse()

#plt.plot(twiss_b1_start['s', : ], twiss_b1_start['dx', : ],label='x')
plt.plot(twiss_b2_start['s', : ], twiss_b2_start['dy', : ],label='y')
plt.axvline(twiss_b2_start[['s'],'ip8'],color = 'green', linestyle='-.', label='IP8')
plt.axvline(twiss_b2_start[['s'],'ip1'],color = 'black', linestyle='-.', label='IP1')
plt.axvline(twiss_b2_start[['s'],'ip2'],color = 'red', linestyle='-.', label='IP2')
plt.title(f'Dispersion')
plt.legend()
plt.grid(True)
# %%
