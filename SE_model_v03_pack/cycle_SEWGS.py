# %%
# Importing
# %%
from pySEWGS import AdsCatColumn
import numpy as np
import os

# %%
# Result storage directory path
# %%
dir_name = 'Sim_results_250204'
os.makedirs(dir_name, exist_ok = True)
# %%
# Adsorbent & Catalyst Conditions

# %%
## 1) Define AdsCatCol
## 2) pac_info: packing info
## 3) ads_info: isotherm func & density
## 4) cat_info: rate constant func & density
L = 2           # Length (m)
N = 41          # Number of spatial node 

epsi = 0.4      # Void fraction (-)
x_cat = 0.7     # Catalyst fraction
d_p = 2E-3      # Particle size

rho_ads = 1000 # kg/m3  //density of adsorbent
rho_cat = 1000 # kg/m3  //density of catalyst

k_list = [0.05,]*4  # Adsorption rate constant (1/s)
k_f_ref = 1E-3      # Kinetic parameter at T_ref : mol / (kg s) / (bar^2)
T_ref = 623         # K
E_a_f = 1E4         # J/mol

qm1,qm2,qm3,qm4 = 0.02, 0.3, 0.6, 3.0
b1, b2, b3, b4  = 0.05, 0.1, 0.3, 0.5
dH1,dH2,dH3,dH4 = [18000,]*4
T_ref1, T_ref2, T_ref3, T_ref4 = [623,]*4 
# H2, CO, H2O, CO2
def Arrh(T, dH, T_ref):
    term_exp = dH/8.3145*(1/T - 1/T_ref)
    theta = np.exp(term_exp)
    return theta

def iso(P1, P2, P3, P4, T): # isotherm function
    the1 = Arrh(T, dH1, T_ref1)
    the2 = Arrh(T, dH2, T_ref2)
    the3 = Arrh(T, dH3, T_ref3)
    the4 = Arrh(T, dH4, T_ref4)
    deno = 1+b1*P1*the1 + b2*P2*the2 + b3*P3*the3 + b4*P4*the4
    nume1 = qm1*b1*P1*the1
    nume2 = qm2*b2*P2*the2
    nume3 = qm3*b3*P3*the3
    nume4 = qm4*b4*P4*the4

    q1 = nume1/deno
    q2 = nume2/deno
    q3 = nume3/deno
    q4 = nume4/deno
    return q1, q2, q3, q4

acc1 = AdsCatColumn(L, N,)
acc1.pac_info(epsi, x_cat, d_p)
acc1.ads_info(iso, k_list, rho_ads, )
acc1.cat_info(k_f_ref, T_ref, E_a_f, rho_cat, )
# %%
# Feed & outlet conditions
# %%
## 5) feed_condi: feed composition & conditions
## 6) outlet_condi: exit composition & conditions
y_feed_list = [0, 0.5, 0.5, 0]
P_feed = 7.0 # bar
T_feed = 623 # K
u_feed = 0.05 # m/s
acc1.feed_condi_y(y_feed_list, P_feed, T_feed, u_feed, )

y_end_list = [1, 0, 0, 0.0]
P_end = 5.5 # bar
Cv_out1 = 2E-2
T_end = 623 # K
acc1.outlet_condi_y(y_end_list, P_end, T_end, Cv_out1, )

# %%
# Initial conditions
# %%
## 7) init_condi: initial conditions
P_init = 5.7*np.ones(N)
T_init = 623*np.ones(N)
y1_i = 1.0*np.ones([N,])
y2_i = 0.0*np.ones([N,])
y3_i = 0.0*np.ones([N,])
y4_i = 0.0*np.ones([N,])
y_init = [y1_i, y2_i, y3_i, y4_i,]
acc1.init_condi_y(P_init, y_init, None, T_init, True)
## 8) Flow direction setting
acc1.Flow_direction(True)
## 9) Cv setting
Cv_Pfeed_succ = acc1.set_Cv_Pfeed(P_mid = 7.0, 
                         x_vel_feed_out = 0.6,)
Cv_end_cal, P_feed_cal, succ = Cv_Pfeed_succ
print('[Adsorption step]')
print('Cv out_cal: ', Cv_end_cal)
print('P_feed: ', P_feed_cal)
acc1.feed_condi_y(y_feed_list, P_feed_cal,
                    T_feed, u_feed,)
acc1.outlet_condi_y(y_end_list, P_end, T_end, 
                    Cv_end_cal,)

# %%
# RUN simulations
# %%
#t_ran = np.arange(0,320+0.0025, 0.0025)
t_ran = np.arange(0,75+0.0025, 0.0025)
acc1.run_mamo(t_ran,)

# %%
# Graph
# %%

f_prefix = 'sim_c01_s01rxn_'

figsize_test = [12.5,5]
t_interval = 2500
ShowGraph_mol = False
# Uptake: H2 & CO2
acc1.graph_still(4, t_frame = t_interval,
                     label = 'H$_{2}$ uptake (mol/kg)',
                     filename=dir_name+'/'+f_prefix+'q1.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)
acc1.graph_still(7, t_frame = t_interval,
                     label = 'CO$_{2}$ uptake (mol/kg)',
                     filename=dir_name+'/'+f_prefix+'q4.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)

# Mole fraction: H2 & CO2
acc1.graph_still_mol(0, t_frame = t_interval,
                     label = 'H$_{2}$ mole frac.',
                     filename=dir_name+'/'+f_prefix+'y1.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)
acc1.graph_still_mol(3, t_frame = t_interval,
                     label = 'CO$_{2}$ mole frac.',
                     filename=dir_name+'/'+f_prefix+'y4.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)

# GIF mole frac: H2 & CO2
acc1.graph_timelapse_y(0,t_frame=1000,
                       label='H$_{2}$ mole frac.',
                        filename =dir_name+'/'+f_prefix+'y1_timelapse.gif',
                        y_limits=[-0.04,1.04], interval = 100)
acc1.graph_timelapse_y(3,t_frame=1000,
                       label='CO$_{2}$ mole frac.',
                       filename =dir_name+'/'+f_prefix+'y4_timelapse.gif',
                       y_limits=[-0.04,1.04], interval = 100)

# %%
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# =-=-  Next Cycle Test  -=-=-=
# =-=-=-=- Blowdown =-=-=-=-=-=
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# %%
# Next initial conditions
# %%
acc1.next_init(Cq_return= False, change_init= True)
# %%
# Feed condition
T_feed2 = 873   # T_feed1 = 623 K (350 dgr C)
u_feed2 = 0     # u_feed1 = 0.05 m/s
y_feed_list2 = [1,0,0,0]
acc1.feed_condi_y(y_feed_list2, 1, T_feed2, u_feed2, 
                  Cv_in= 0.02, const_u = True,)
# Outlet condition
P_end2 = 1.1    # P_end1 = 5.5 (bar)
Cv_out2 = 0.08  # Cv_out1 = 0.02
y_end_list2 = [1,0,0,0]
acc1.outlet_condi_y(y_end_list2, P_end2,
                    T_end, Cv_out2)

# %%
# Run the Simulations
# %%
acc1.Flow_direction(False)
t_ran_bl = np.arange(0, 50+0.0025, 0.0025)
acc1.run_mamo(t_ran_bl)

# %%
# Grpah for Blowdown step
# %%
f_prefix = 'sim_c01_s02blo_'
os.makedirs(dir_name, exist_ok = True)
figsize_test = [12.5,5]
t_interval = 2500
ShowGraph_mol = False
acc1.graph_still('P', t_frame = t_interval,
                     label = 'Pressure (bar)',
                     filename=dir_name+'/'+f_prefix+'P.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)

# Uptake: H2 & CO2
acc1.graph_still(4, t_frame = t_interval,
                     label = 'H$_{2}$ uptake (mol/kg)',
                     filename=dir_name+'/'+f_prefix+'q1.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)
acc1.graph_still(7, t_frame = t_interval,
                     label = 'CO$_{2}$ uptake (mol/kg)',
                     filename=dir_name+'/'+f_prefix+'q4.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)

# Mole fraction: H2 & CO2
acc1.graph_still_mol(0, t_frame = t_interval,
                     label = 'H$_{2}$ mole frac.',
                     filename=dir_name+'/'+f_prefix+'y1.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)
acc1.graph_still_mol(3, t_frame = t_interval,
                     label = 'CO$_{2}$ mole frac.',
                     filename=dir_name+'/'+f_prefix+'y4.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)

# GIF mole frac: H2 & CO2
acc1.graph_timelapse_y(0,t_frame=1000,
                       label='H$_{2}$ mole frac.',
                        filename =dir_name+'/'+f_prefix+'y1_timelapse.gif',
                        y_limits=[-0.04,1.04], interval = 100)
acc1.graph_timelapse_y(3,t_frame=1000,
                       label='CO$_{2}$ mole frac.',
                       filename =dir_name+'/'+f_prefix+'y4_timelapse.gif',
                       y_limits=[-0.04,1.04], interval = 100)
acc1.graph_timelapse_y('P',t_frame=1000,
                       label='H$_{2}$ mole frac.',
                        filename =dir_name+'/'+f_prefix+'P_timelapse.gif',
                        y_limits=[0.9,7.5], interval = 100)

# %%

# %%
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# =-=-  Next Cycle Test  -=-=-=
# =-=-=-=- Purge =-=-=-=-=-=
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# %%
# Next initial conditions
# %%
acc1.next_init(Cq_return= False, change_init= True)
# %%
# Feed condition
y_feed_list3 = [1, 0, 0, 0]
P_feed3 = 1.9
T_feed3 = 873   # T_feed1 = 623 K (350 dgr C)
u_feed3 = 0.05     # u_feed1 = 0.05 m/s

acc1.feed_condi_y(y_feed_list3, P_feed3, T_feed3, u_feed3, 
                  Cv_in= 0.04, const_u = True,)
# Outlet condition
P_end3 = 1.1    # P_end1 = 5.5 (bar)
T_end3 = 873
Cv_out3 = 0.08  # Cv_out1 = 0.02
y_end_list3 = [1,0,0,0]
acc1.outlet_condi_y(y_end_list3, P_end3,
                    T_end, Cv_out3)

## 9) Cv setting
Cv_Pfeed_succ = acc1.set_Cv_Pfeed(P_mid = 1.5, 
                         x_vel_feed_out = 0.8,)
Cv_end_cal3, P_feed_cal3, succ3 = Cv_Pfeed_succ
print('[Purge step]')
print('Cv out_cal: ', Cv_end_cal3)
print('P_feed: ', P_feed_cal3)
acc1.feed_condi_y(y_feed_list3, P_feed_cal3,
                    T_feed3, u_feed3,)
acc1.outlet_condi_y(y_end_list3, P_end3, T_end3, 
                    Cv_end_cal3,)

# %%
# Run the Simulations
# %%
acc1.Flow_direction(False)
t_ran_pu = np.arange(0, 75+0.0025, 0.0025)
acc1.run_mamo(t_ran_bl)

# %%
# Grpah for Blowdown step
# %%

f_prefix = 'sim_c01_s03pur_'
os.makedirs(dir_name, exist_ok = True)
figsize_test = [12.5,5]
t_interval = 2500
ShowGraph_mol = False
# Pressure
acc1.graph_still_mol('P', t_frame = t_interval,
                     label = 'Pressure (bar)',
                     filename=dir_name+'/'+f_prefix+'P.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)

# Uptake: H2 & CO2
acc1.graph_still(4, t_frame = t_interval,
                     label = 'H$_{2}$ uptake (mol/kg)',
                     filename=dir_name+'/'+f_prefix+'q1.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)
acc1.graph_still(7, t_frame = t_interval,
                     label = 'CO$_{2}$ uptake (mol/kg)',
                     filename=dir_name+'/'+f_prefix+'q4.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)

# Mole fraction: H2 & CO2
acc1.graph_still_mol(0, t_frame = t_interval,
                     label = 'H$_{2}$ mole frac.',
                     filename=dir_name+'/'+f_prefix+'y1.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)
acc1.graph_still_mol(3, t_frame = t_interval,
                     label = 'CO$_{2}$ mole frac.',
                     filename=dir_name+'/'+f_prefix+'y4.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)


# GIF mole frac: H2 & CO2
acc1.graph_timelapse_y(0,t_frame=1000,
                       label='H$_{2}$ mole frac.',
                        filename =dir_name+'/'+f_prefix+'y1_timelapse.gif',
                        y_limits=[-0.04,1.04], interval = 100)
acc1.graph_timelapse_y(3,t_frame=1000,
                       label='CO$_{2}$ mole frac.',
                       filename =dir_name+'/'+f_prefix+'y4_timelapse.gif',
                       y_limits=[-0.04,1.04], interval = 100)
acc1.graph_timelapse_y('P',t_frame=1000,
                       label='Pressure (bar)',
                       filename =dir_name+'/'+f_prefix+'P_timelapse.gif',
                       y_limits=[0.9,7.5], interval = 100)


# %%
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# =-=-  Next Cycle Test  -=-=-=
# =-=-= Pressurization  =-=-=-=
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# %%
# Next initial conditions
# %%
acc1.next_init(Cq_return= False, change_init= True)
# %%
# Feed condition
y_feed_list4 = [0,.5,.5,0]
P_feed4 = 6.9
T_feed4 = 623   # T_feed1 = 623 K (350 dgr C)
u_feed4 = 0.01   # u_feed1 = 0.05 m/s

acc1.feed_condi_y(y_feed_list4, P_feed4, T_feed4, u_feed4, 
                  Cv_in= 0.05, const_u = False,)
# Outlet condition
P_end4 = 5.5    # P_end1 = 5.5 (bar)
Cv_out4 = 0.00  # Cv_out1 = 0.02
T_end4 = 623
y_end_list4 = [1,0,0,0]
acc1.outlet_condi_y(y_end_list4, P_end4,
                    T_end4, Cv_out4)

# %%
# Run the Simulations
# %%
acc1.Flow_direction(True)
t_ran_pr = np.arange(0, 50+0.0025, 0.0025)
acc1.run_mamo(t_ran_pr)

# %%
# Grpah for Blowdown step
# %%
f_prefix = 'sim_c01_s04pre_'
figsize_test = [12.5,5]
t_interval = 2500
ShowGraph_mol = True
# Pressure
acc1.graph_still('P', t_frame = t_interval,
                     label = 'Pressure (bar)',
                     filename=dir_name+'/'+f_prefix+'P.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)

# Uptake: H2 & CO2
acc1.graph_still(4, t_frame = t_interval,
                     label = 'H$_{2}$ uptake (mol/kg)',
                     filename=dir_name+'/'+f_prefix+'q1.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)
acc1.graph_still(7, t_frame = t_interval,
                     label = 'CO$_{2}$ uptake (mol/kg)',
                     filename=dir_name+'/'+f_prefix+'q4.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)

# Mole fraction: H2 & CO2
acc1.graph_still_mol(0, t_frame = t_interval,
                     label = 'H$_{2}$ mole frac.',
                     filename=dir_name+'/'+f_prefix+'y1.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)
acc1.graph_still_mol(3, t_frame = t_interval,
                     label = 'CO$_{2}$ mole frac.',
                     filename=dir_name+'/'+f_prefix+'y4.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)

# GIF mole frac: H2 & CO2
acc1.graph_timelapse_y('P',t_frame=1000,
                       label='H$_{2}$ mole frac.',
                        filename =dir_name+'/'+f_prefix+'P_timelapse.gif',
                        y_limits=[0.9,7.5], interval = 100)
acc1.graph_timelapse_y(0,t_frame=1000,
                       label='H$_{2}$ mole frac.',
                        filename =dir_name+'/'+f_prefix+'y1_timelapse.gif',
                        y_limits=[-0.04,1.04], interval = 100)
acc1.graph_timelapse_y(3,t_frame=1000,
                       label='CO$_{2}$ mole frac.',
                       filename =dir_name+'/'+f_prefix+'y4_timelapse.gif',
                       y_limits=[-0.04,1.04], interval = 100)

# %%