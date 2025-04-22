# %%
# Importing
# %%
from pySEWGS import AdsCatColumn
import numpy as np
from numpy import trapz
import os
import matplotlib.pyplot as plt
from datetime import datetime
import time
import math


# %%
# 4/21 -> x_cat = 0.6/ 0.2 => 0.4 uniform
# 4/21 -> k_f_ref = 1E-3 => 3E-3
# 4/21 -> k_list = [0.5,]*4  => [0.8,]*4
# 4/22 -> k_list = [0.8,]*4 => [1.5,]*4
# %%
# Adsorbent & Catalyst Conditions
# %%
## 1) Define AdsCatCol
## 2) pac_info: packing info
## 3) ads_info: isotherm func & density
## 4) cat_info: rate constant func & density
L = 1.2           # Length (m)
N = 51          # Number of spatial node 

z_dom = np.linspace(0,L,N)
R_gas = 8.3145


#x_cat = 0.2     # Catalyst fraction
# 조건에 맞는 배열 생성
#x_cat_array = np.zeros(N)

#x_cat_array[:20] = 0.6  # 앞쪽 20개는 0.6
#x_cat_array[20] = 0.4  # 중간 노드 하나는 0.4
#x_cat_array[21:] = 0.2  # 뒤쪽 20개는 0.2
#x_cat = x_cat_array

ratio_front = 0.4
ratio_back = 0.4
xcat_avg = 0.4


n_front = N // 2         # 20개  0-20
n_back = N - n_front     # 21개  20-40

x_cat_array = np.zeros(N)
x_cat_array[:n_front] = ratio_front
x_cat_array[n_front:] = ratio_back

current_avg = np.mean(x_cat_array)

correction = xcat_avg / current_avg

# 전체 평균 맞추도록 비율 유지하면서 보정
x_cat_array_correct = x_cat_array * correction
print(f"전체 평균 (target): {np.mean(x_cat_array_correct):.4f}")


x_cat = x_cat_array_correct


epsi = 0.4      # Void fraction (-)
d_p = 2E-3      # Particle size

rho_ads = 1000 # kg/m3  //density of adsorbent
rho_cat = 1000 # kg/m3  //density of catalyst

#k_list = [0.05,]*4
k_list = [0.8,]*4  # Adsorption rate constant (1/s)
# Previously k_list = [0.5,]*4
k_f_ref = 3E-3      # Kinetic parameter at T_ref : mol / (kg s) / (bar^2)
# Previously k_f_ref = 1E-3
T_ref = 623         # K
E_a_f = 1E4         # J/mol

#qm1,qm2,qm3,qm4 = 0.02, 0.3, 0.6, 3.0
# H2, CO, H2O, CO2
qm1,qm2,qm3,qm4 = 0.02, 0.2, 0.2, 3.0
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
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# =-=- Cyclic st.st Test -=-=-=
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Cyclic Steady State

dir_name1 = 'Sim_results01_feed'
os.makedirs(dir_name1, exist_ok = True)
dir_name2 = 'Sim_results02_dep'
os.makedirs(dir_name2, exist_ok = True)
dir_name3 = 'Sim_results03_purge'
os.makedirs(dir_name3, exist_ok = True)
dir_name4 = 'Sim_results04_rep'
os.makedirs(dir_name4, exist_ok = True)

dir_name1 = 'Sim_results01_feed'
dir_name2 = 'Sim_results02_dep'
dir_name3 = 'Sim_results03_purge'
dir_name4 = 'Sim_results04_rep'

for dir_name in [dir_name1, dir_name2, dir_name3, dir_name4]:
    os.makedirs(dir_name, exist_ok=True)

#T_feed = T_end - (T_end-T_start)*np.exp(-t/tau) [:, np.newaxis]
t_ran1 = np.arange(0,75+0.0025,0.0025)
t_ran2 = np.arange(0,50+0.0025,0.0025)
tau1 = t_ran1[-1]/3
tau2 = t_ran2[-1]/3

A_cros = np.pi / 4 * (0.2) ** 2  #cros-sectional area
cycle_N = 20 # number of cycle



CO_conversion_list = []
CO_conversion_list_feed = []
H2_purity_feed_list = []
H2_purity_H2OX_list = []

molfrac_H2_fd_list = []
molfrac_CO_fd_list = []
molfrac_H2O_fd_list = []
molfrac_CO2_fd_list = []

molfrac_H2_dep_list = []
molfrac_CO_dep_list = []
molfrac_H2O_dep_list = []
molfrac_CO2_dep_list = []

molfrac_H2_pur_list = []
molfrac_CO_pur_list = []
molfrac_H2O_pur_list = []
molfrac_CO2_pur_list = []



P_init = 5.5*np.ones(N)
T_init = 623*np.ones(N)
y1_i = 1.0*np.ones([N,])
y2_i = 0.0*np.ones([N,])
y3_i = 0.0*np.ones([N,])
y4_i = 0.0*np.ones([N,])
y_init = [y1_i, y2_i, y3_i, y4_i,]

T_feed = 623  # 623K = 350 K
T_feed2 = 623 # 623K = 350 K
T_feed3 = 623 # 623K = 350 K
T_feed4 = 623 # 623K = 350 K

start_time = time.time()

for i in range(0, cycle_N):

    f_prefix1 = 'sim_s01feed_'
    f_prefix2 = 'sim_s02dep_'
    f_prefix3 = 'sim_s03purge_'
    f_prefix4 = 'sim_s04rep_'
    f_prefix1g = 'gif_sim_s01feed_'
    f_prefix2g = 'gif_sim_s02dep_'
    f_prefix3g = 'gif_sim_s03purge_'
    f_prefix4g = 'gif_sim_s04rep_'




    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # step1 Feed
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('')
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    print(f'This simulation is No.{i+1} Feed')
    print(datetime.now())
    before_all = time.time()
    before = time.time()

    if i == 0:
        P_init = 5.5*np.ones(N)
        T_init = 623*np.ones(N)
        y1_i = 1.0*np.ones([N,])
        y2_i = 0.0*np.ones([N,])
        y3_i = 0.0*np.ones([N,])
        y4_i = 0.0*np.ones([N,])
        y_init = [y1_i, y2_i, y3_i, y4_i,]
        acc1.init_condi_y(P_init, y_init, None, T_init, True)

    else:
        acc1.next_init(Cq_return= False, change_init= True) 
    
    y_feed_list = [0, 0.3, 0.7, 0]
    P_feed = 6.0 # bar
    T_feed = 623 # K
    u_feed = 0.05 # m/s
    acc1.feed_condi_y(y_feed_list, P_feed, T_feed, u_feed, )
    y_end_list = [1, 0, 0, 0.0]
    P_end = 5.0 # bar
    Cv_out1 = 5E-2
    T_end = 623 # K
    acc1.outlet_condi_y(y_end_list, P_end, T_end, Cv_out1, )
    
    acc1.Flow_direction(True)
    Cv_Pfeed_succ = acc1.set_Cv_Pfeed(P_mid = 6, 
                            x_vel_feed_out = 0.7,)
    Cv_end_cal, P_feed_cal, succ = Cv_Pfeed_succ
    print('[Feed step]')
    print('Cv out_cal: ', Cv_end_cal)
    print('P_feed: ', P_feed_cal)
    acc1.feed_condi_y(y_feed_list, P_feed_cal,
                        T_feed, u_feed,)
    acc1.outlet_condi_y(y_end_list, P_end, T_end, 
                Cv_end_cal,)
    
    acc1.run_mamo(t_ran1)

    after = time.time()
    print(f'It takes {math.floor(after - before)} sec')
    print('')
    
    figsize_test = [12.5,5]
    t_interval = 2000
    ShowGraph_mol = False
    acc1.graph_still('P', t_frame = t_interval,
                        label = f'No.{i+1} Pressure (bar)',
                        filename=dir_name1+'/'+f_prefix1+f'P_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)

    # Uptake: H2 & CO2
    acc1.graph_still(4, t_frame = t_interval,
                        label = f'No.{i+1} H$_{2}$ uptake (mol/kg)',
                        filename=dir_name1+'/'+f_prefix1+f'q1_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still(7, t_frame = t_interval,
                        label = f'No.{i+1} CO$_{2}$ uptake (mol/kg)',
                        filename=dir_name1+'/'+f_prefix1+f'q4_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    
    # Concentration: H2 & CO2
    acc1.graph_still(0, t_frame = t_interval,
                        label = f'No.{i+1} H$_{2}$ Concentration (mol/m$^{3}$)',
                        filename=dir_name1+'/'+f_prefix1+f'c1_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still(1, t_frame = t_interval,
                        label = f'No.{i+1} CO Concentration (mol/m$^{3}$)',
                        filename=dir_name1+'/'+f_prefix1+f'c2_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still(2, t_frame = t_interval,
                        label = f'No.{i+1} H$_{2}$O Concentration (mol/m$^{3}$)',
                        filename=dir_name1+'/'+f_prefix1+f'c3_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still(3, t_frame = t_interval,
                        label = f'No.{i+1} CO$_{2}$ Concentration (mol/m$^{3}$)',
                        filename=dir_name1+'/'+f_prefix1+f'c4_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)

    # Mole fraction: H2 & CO2
    acc1.graph_still_mol(0, t_frame = t_interval,
                        label = f'No.{i+1} H$_{2}$ mole frac.',
                        filename=dir_name1+'/'+f_prefix1+f'y1_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still_mol(1, t_frame = t_interval,
                        label = f'No.{i+1} CO mole frac.',
                        filename=dir_name1+'/'+f_prefix1+f'y2_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still_mol(3, t_frame = t_interval,
                        label = f'No.{i+1} CO$_{2}$ mole frac.',
                        filename=dir_name1+'/'+f_prefix1+f'y4_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    
    #Gif making
    if i+1 in [1,2,3,4,5,10,20, 25, 50]:
        acc1.graph_timelapse(0,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$ Concentration (mol/m$^{3}$)',
                            filename =dir_name1+'/'+f_prefix1g+f'c1_cycle{i+1}.gif',
                            y_limits=[-5,120], interval = 100)
        acc1.graph_timelapse(1,t_frame=1000,
                            label=f'No.{i+1} CO Concentration (mol/m$^{3}$)',
                            filename =dir_name1+'/'+f_prefix1g+f'c2_cycle{i+1}.gif',
                            y_limits=[-5,80], interval = 100)
        acc1.graph_timelapse(2,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$O Concentration (mol/m$^{3}$)',
                            filename =dir_name1+'/'+f_prefix1g+f'c3_cycle{i+1}.gif',
                            y_limits=[-5,80], interval = 100)
        acc1.graph_timelapse(3,t_frame=1000,
                            label=f'No.{i+1} CO$_{2}$ Concentration (mol/m$^{3}$)',
                            filename =dir_name1+'/'+f_prefix1g+f'c4_cycle{i+1}.gif',
                            y_limits=[-5,30], interval = 100)
        
        acc1.graph_timelapse(4,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$ uptake (mol/kg)',
                            filename =dir_name1+'/'+f_prefix1g+f'q1_cycle{i+1}.gif',
                            y_limits=[-0.0005,0.005], interval = 100)
        acc1.graph_timelapse(5,t_frame=1000,
                            label=f'No.{i+1} CO uptake (mol/kg)',
                            filename =dir_name1+'/'+f_prefix1g+f'q2_cycle{i+1}.gif',
                            y_limits=[-0.005,0.2], interval = 100)
        acc1.graph_timelapse(6,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$O uptake (mol/kg)',
                            filename =dir_name1+'/'+f_prefix1g+f'q3_cycle{i+1}.gif',
                            y_limits=[-0.005,0.2], interval = 100)
        acc1.graph_timelapse(7,t_frame=1000,
                            label=f'No.{i+1} CO$_{2}$ uptake (mol/kg)',
                            filename =dir_name1+'/'+f_prefix1g+f'q4_cycle{i+1}.gif',
                            y_limits=[-0.005,0.2], interval = 100)
        
        acc1.graph_timelapse_y('P',t_frame=1000,
                            label=f'No.{i+1} Pressure',
                            filename =dir_name1+'/'+f_prefix1g+f'P_cycle{i+1}.gif',
                            y_limits=[0.9,7.5], interval = 100)
        acc1.graph_timelapse_y(0,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$ mole frac.',
                            filename =dir_name1+'/'+f_prefix1g+f'y1_cycle{i+1}.gif',
                            y_limits=[-0.04,1.04], interval = 100)
        acc1.graph_timelapse_y(1,t_frame=1000,
                            label=f'No.{i+1} CO mole frac.',
                            filename =dir_name1+'/'+f_prefix1g+f'y2_cycle{i+1}.gif',
                            y_limits=[-0.04,1.04], interval = 100)
        acc1.graph_timelapse_y(2,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$O mole frac.',
                            filename =dir_name1+'/'+f_prefix1g+f'y3_cycle{i+1}.gif',
                            y_limits=[-0.04,1.04], interval = 100)
        acc1.graph_timelapse_y(3,t_frame=1000,
                            label=f'No.{i+1} CO$_{2}$ mole frac.',
                            filename =dir_name1+'/'+f_prefix1g+f'y4_cycle{i+1}.gif',
                            y_limits=[-0.04,1.04], interval = 100)

    TP_fd1 = T_feed - (T_feed - T_feed4)*np.exp(-t_ran1/tau1)
    TP_feed1 = np.broadcast_to(TP_fd1[:, np.newaxis], (30001, N))

    C_ov_step1 = acc1.y_res[:, 0*N:1*N] + acc1.y_res[:, 1*N:2*N] + acc1.y_res[:, 2*N:3*N] + acc1.y_res[:, 3*N:4*N]
    P_ov_step1_T = C_ov_step1*R_gas*TP_feed1/1E5

    C2_feed_fd = y_feed_list[1]*P_feed_cal*1E5/R_gas/T_feed
    CO_in_fd = u_feed*C2_feed_fd*A_cros*epsi
    inte_CO_in_fd = trapz(CO_in_fd * np.ones_like(t_ran1), t_ran1)*2

    #out
    P_out_fd = P_ov_step1_T[:,-1]
    u_out_fd = Cv_end_cal*(P_out_fd-P_end)
    
    CO_out_fd = u_out_fd*acc1.y_res[:, 1*N:2*N][:,-1]*A_cros*epsi
    inte_CO_out_fd = trapz(CO_out_fd, t_ran1)*2

    #feed all gas out
    all_out_fd = u_out_fd*C_ov_step1[:,-1]*A_cros*epsi
    inte_out_fd = trapz(all_out_fd, t_ran1)*2

    H2_out_fd = u_out_fd*acc1.y_res[:, 0*N:1*N][:,-1]*A_cros*epsi
    inte_H2_out_fd = trapz(H2_out_fd, t_ran1)*2

    H2O_out_fd = u_out_fd*acc1.y_res[:, 2*N:3*N][:,-1]*A_cros*epsi
    inte_H2O_out_fd = trapz(H2O_out_fd, t_ran1)*2

    CO2_out_fd = u_out_fd*acc1.y_res[:, 3*N:4*N][:,-1]*A_cros*epsi
    inte_CO2_out_fd = trapz(CO2_out_fd, t_ran1)*2



    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # step2 Depressurization
    print(f'This simulation is No.{i+1} Depressurization')
    print(datetime.now())
    before = time.time()
    
    acc1.next_init(Cq_return=True, change_init=True)

    T_feed2 = 623   # T_feed1 = 623 K (350 dgr C)
    #acc1.next_init_p(T_feed, T_feed2, Cq_return = False, change_init = True)
    # Feed condition
    u_feed2 = 0     # u_feed1 = 0.05 m/s
    P_feed2 = 8.5
    y_feed_list2 = [1,0,0,0]
    acc1.feed_condi_y(y_feed_list2, P_feed2, T_feed2, u_feed2, 
                    Cv_in= 0.02, const_u = True,)
    # Outlet condition
    T_end2 = 623
    P_end2 = 1.0    # P_end1 = 5.5 (bar)
    Cv_out2 = 8E-2  # Cv_out1 = 0.02
    y_end_list2 = [1,0,0,0]
    acc1.outlet_condi_y(y_end_list2, P_end2,
                        T_end2, Cv_out2)

    acc1.Flow_direction(False)
    acc1.run_mamo(t_ran2)
    
    after = time.time()
    print(f'It takes {math.floor(after - before)} sec')
    print('')

    figsize_test = [12.5,5]
    t_interval = 2000
    ShowGraph_mol = False
    acc1.graph_still('P', t_frame = t_interval,
                        label = f'No.{i+1} Pressure (bar)',
                        filename=dir_name2+'/'+f_prefix2+f'P_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)

    # Uptake: H2 & CO2
    acc1.graph_still(4, t_frame = t_interval,
                        label = f'No.{i+1} H$_{2}$ uptake (mol/kg)',
                        filename=dir_name2+'/'+f_prefix2+f'q1_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still(7, t_frame = t_interval,
                        label = f'No.{i+1} CO$_{2}$ uptake (mol/kg)',
                        filename=dir_name2+'/'+f_prefix2+f'q4_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)

    # Concentration: H2 & CO2
    acc1.graph_still(0, t_frame = t_interval,
                        label = f'No.{i+1} H$_{2}$ Concentration (mol/m$^{3}$)',
                        filename=dir_name2+'/'+f_prefix2+f'c1_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still(1, t_frame = t_interval,
                        label = f'No.{i+1} CO Concentration (mol/m$^{3}$)',
                        filename=dir_name2+'/'+f_prefix2+f'c2_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still(2, t_frame = t_interval,
                        label = f'No.{i+1} H$_{2}$O Concentration (mol/m$^{3}$)',
                        filename=dir_name2+'/'+f_prefix2+f'c3_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still(3, t_frame = t_interval,
                        label = f'No.{i+1} CO$_{2}$ Concentration (mol/m$^{3}$)',
                        filename=dir_name2+'/'+f_prefix2+f'c4_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)

    # Mole fraction: H2 & CO2
    acc1.graph_still_mol(0, t_frame = t_interval,
                        label = f'No.{i+1} H$_{2}$ mole frac.',
                        filename=dir_name2+'/'+f_prefix2+f'y1_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still_mol(1, t_frame = t_interval,
                        label = f'No.{i+1} CO mole frac.',
                        filename=dir_name2+'/'+f_prefix2+f'y2_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still_mol(3, t_frame = t_interval,
                        label = f'No.{i+1} CO$_{2}$ mole frac.',
                        filename=dir_name2+'/'+f_prefix2+f'y4_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)

    #Gif making
    if i+1 in [1, 2, 3, 4, 5, 10, 20, 25, 50]:
        acc1.graph_timelapse(0,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$ Concentration (mol/m$^{3}$)',
                            filename =dir_name2+'/'+f_prefix2g+f'c1_cycle{i+1}.gif',
                            y_limits=[-5,120], interval = 100)
        acc1.graph_timelapse(1,t_frame=1000,
                            label=f'No.{i+1} CO Concentration (mol/m$^{3}$)',
                            filename =dir_name2+'/'+f_prefix2g+f'c2_cycle{i+1}.gif',
                            y_limits=[-5,80], interval = 100)
        acc1.graph_timelapse(2,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$O Concentration (mol/m$^{3}$)',
                            filename =dir_name2+'/'+f_prefix2g+f'c3_cycle{i+1}.gif',
                            y_limits=[-5,80], interval = 100)
        acc1.graph_timelapse(3,t_frame=1000,
                            label=f'No.{i+1} CO$_{2}$ Concentration (mol/m$^{3}$)',
                            filename =dir_name2+'/'+f_prefix2g+f'c4_cycle{i+1}.gif',
                            y_limits=[-5,30], interval = 100)
        
        acc1.graph_timelapse(4,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$ uptake (mol/kg)',
                            filename =dir_name2+'/'+f_prefix2g+f'q1_cycle{i+1}.gif',
                            y_limits=[-0.0005,0.005], interval = 100)
        acc1.graph_timelapse(5,t_frame=1000,
                            label=f'No.{i+1} CO uptake (mol/kg)',
                            filename =dir_name2+'/'+f_prefix2g+f'q2_cycle{i+1}.gif',
                            y_limits=[-0.005,0.2], interval = 100)
        acc1.graph_timelapse(6,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$O uptake (mol/kg)',
                            filename =dir_name2+'/'+f_prefix2g+f'q3_cycle{i+1}.gif',
                            y_limits=[-0.005,0.2], interval = 100)
        acc1.graph_timelapse(7,t_frame=1000,
                            label=f'No.{i+1} CO$_{2}$ uptake (mol/kg)',
                            filename =dir_name2+'/'+f_prefix2g+f'q4_cycle{i+1}.gif',
                            y_limits=[-0.005,0.2], interval = 100)
        
        acc1.graph_timelapse_y('P',t_frame=1000,
                            label=f'No.{i+1} Pressure',
                            filename =dir_name2+'/'+f_prefix2g+f'P_cycle{i+1}.gif',
                            y_limits=[0.9,7.5], interval = 100)
        acc1.graph_timelapse_y(0,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$ mole frac.',
                            filename =dir_name2+'/'+f_prefix2g+f'y1_tcycle{i+1}.gif',
                            y_limits=[-0.04,1.04], interval = 100)
        acc1.graph_timelapse_y(1,t_frame=1000,
                            label=f'No.{i+1} CO mole frac.',
                            filename =dir_name2+'/'+f_prefix2g+f'y2_cycle{i+1}.gif',
                            y_limits=[-0.04,1.04], interval = 100)
        acc1.graph_timelapse_y(2,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$O mole frac.',
                            filename =dir_name2+'/'+f_prefix2g+f'y3_cycle{i+1}.gif',
                            y_limits=[-0.04,1.04], interval = 100)
        acc1.graph_timelapse_y(3,t_frame=1000,
                            label=f'No.{i+1} CO$_{2}$ mole frac.',
                            filename =dir_name2+'/'+f_prefix2g+f'y4_cycle{i+1}.gif',
                            y_limits=[-0.04,1.04], interval = 100)
    

    TP_fd2 = T_feed2 - (T_feed2 - T_feed)*np.exp(-t_ran2/tau2)
    TP_feed2 = np.broadcast_to(TP_fd2[:, np.newaxis], (20001, N))

    C_ov_step2 = acc1.y_res[:, 0*N:1*N] + acc1.y_res[:, 1*N:2*N] + acc1.y_res[:, 2*N:3*N] + acc1.y_res[:, 3*N:4*N]
    P_ov_step2_T = C_ov_step2*R_gas*TP_feed2/1E5

    #dep out
    P_out_dep = P_ov_step2_T[:,0]
    u_out_dep = Cv_out2*(P_out_dep-P_end2)

    #dep all gas out
    all_out_dep = u_out_dep*C_ov_step2[:,0]*A_cros*epsi
    inte_out_dep = trapz(all_out_dep, t_ran2)*2

    H2_out_dep = u_out_dep*acc1.y_res[:, 0*N:1*N][:,0]*A_cros*epsi
    inte_H2_out_dep = trapz(H2_out_dep, t_ran2)*2

    CO_out_dep = u_out_dep*acc1.y_res[:, 1*N:2*N][:,0]*A_cros*epsi
    inte_CO_out_dep = trapz(CO_out_dep, t_ran2)*2

    H2O_out_dep = u_out_dep*acc1.y_res[:, 2*N:3*N][:,0]*A_cros*epsi
    inte_H2O_out_dep = trapz(H2O_out_dep, t_ran2)*2

    CO2_out_dep = u_out_dep*acc1.y_res[:, 3*N:4*N][:,0]*A_cros*epsi
    inte_CO2_out_dep = trapz(CO2_out_dep, t_ran2)*2


    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # step3 Purge
    print(f'This simulation is No.{i+1} Purge')
    print(datetime.now())
    before = time.time()
    
    acc1.next_init(Cq_return= False, change_init= True)
    # Feed condition
    y_feed_list3 = [1, 0, 0, 0]
    P_feed3 = 2.5
    T_feed3 = 623   # T_feed1 = 623 K (350 dgr C)
    u_feed3 = 0.04     # u_feed1 = 0.05 m/s

    acc1.feed_condi_y(y_feed_list3, P_feed3, T_feed3, u_feed3, 
                    Cv_in= 0.01, const_u = True,)
    # Outlet condition
    P_end3 = 0.6    # P_end1 = 5.5 (bar)
    T_end3 = 623
    Cv_out3 = 0.02  # Cv_out1 = 0.02
    y_end_list3 = [1,0,0,0]
    acc1.outlet_condi_y(y_end_list3, P_end3,
                        T_end3, Cv_out3)

    ## 9) Cv setting
    Cv_Pfeed_succ = acc1.set_Cv_Pfeed(P_mid = 1.9, 
                            x_vel_feed_out = 2.0,)
    Cv_end_cal3, P_feed_cal3, succ3 = Cv_Pfeed_succ
    print('[Purge step]')
    print('Cv out_cal: ', Cv_end_cal3)
    print('P_feed: ', P_feed_cal3)
    acc1.feed_condi_y(y_feed_list3, P_feed_cal3,
                        T_feed3, u_feed3,)
    acc1.outlet_condi_y(y_end_list3, P_end3, T_end3, 
                        Cv_end_cal3,)
    acc1.Flow_direction(False)
    acc1.run_mamo(t_ran1)

    after = time.time()
    print(f'It takes {math.floor(after - before)} sec')
    print('')


    figsize_test = [12.5,5]
    t_interval = 2000
    ShowGraph_mol = False
    acc1.graph_still('P', t_frame = t_interval,
                        label = f'No.{i+1} Pressure (bar)',
                        filename=dir_name3+'/'+f_prefix3+f'P_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)

    # Uptake: H2 & CO2
    acc1.graph_still(4, t_frame = t_interval,
                        label = f'No.{i+1} H$_{2}$ uptake (mol/kg)',
                        filename=dir_name3+'/'+f_prefix3+f'q1_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still(7, t_frame = t_interval,
                        label = f'No.{i+1} CO$_{2}$ uptake (mol/kg)',
                        filename=dir_name3+'/'+f_prefix3+f'q4_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    
    # Concentration: H2 & CO2
    acc1.graph_still(0, t_frame = t_interval,
                        label = f'No.{i+1} H$_{2}$ Concentration (mol/m$^{3}$)',
                        filename=dir_name3+'/'+f_prefix3+f'c1_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still(1, t_frame = t_interval,
                        label = f'No.{i+1} CO Concentration (mol/m$^{3}$)',
                        filename=dir_name3+'/'+f_prefix3+f'c2_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still(2, t_frame = t_interval,
                        label = f'No.{i+1} H$_{2}$O Concentration (mol/m$^{3}$)',
                        filename=dir_name3+'/'+f_prefix3+f'c3_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still(3, t_frame = t_interval,
                        label = f'No.{i+1} CO$_{2}$ Concentration (mol/m$^{3}$)',
                        filename=dir_name3+'/'+f_prefix3+f'c4_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)

    # Mole fraction: H2 & CO2
    acc1.graph_still_mol(0, t_frame = t_interval,
                        label = f'No.{i+1} H$_{2}$ mole frac.',
                        filename=dir_name3+'/'+f_prefix3+f'y1_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still_mol(1, t_frame = t_interval,
                        label = f'No.{i+1} CO mole frac.',
                        filename=dir_name3+'/'+f_prefix3+f'y2_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still_mol(3, t_frame = t_interval,
                        label = f'No.{i+1} CO$_{2}$ mole frac.',
                        filename=dir_name3+'/'+f_prefix3+f'y4_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)

    #Gif making
    if i+1 in [1, 25, 50]:
        acc1.graph_timelapse(0,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$ Concentration (mol/m$^{3}$)',
                            filename =dir_name3+'/'+f_prefix3g+f'c1_cycle{i+1}.gif',
                            y_limits=[-5,120], interval = 100)
        acc1.graph_timelapse(1,t_frame=1000,
                            label=f'No.{i+1} CO Concentration (mol/m$^{3}$)',
                            filename =dir_name3+'/'+f_prefix3g+f'c2_cycle{i+1}.gif',
                            y_limits=[-5,80], interval = 100)
        acc1.graph_timelapse(2,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$O Concentration (mol/m$^{3}$)',
                            filename =dir_name3+'/'+f_prefix3g+f'c3_cycle{i+1}.gif',
                            y_limits=[-5,80], interval = 100)
        acc1.graph_timelapse(3,t_frame=1000,
                            label=f'No.{i+1} CO$_{2}$ Concentration (mol/m$^{3}$)',
                            filename =dir_name3+'/'+f_prefix3g+f'c4_cycle{i+1}.gif',
                            y_limits=[-5,30], interval = 100)
        
        acc1.graph_timelapse(4,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$ uptake (mol/kg)',
                            filename =dir_name3+'/'+f_prefix3g+f'q1_cycle{i+1}.gif',
                            y_limits=[-0.0005,0.005], interval = 100)
        acc1.graph_timelapse(5,t_frame=1000,
                            label=f'No.{i+1} CO uptake (mol/kg)',
                            filename =dir_name3+'/'+f_prefix3g+f'q2_cycle{i+1}.gif',
                            y_limits=[-0.005,0.2], interval = 100)
        acc1.graph_timelapse(6,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$O uptake (mol/kg)',
                            filename =dir_name3+'/'+f_prefix3g+f'q3_cycle{i+1}.gif',
                            y_limits=[-0.005,0.2], interval = 100)
        acc1.graph_timelapse(7,t_frame=1000,
                            label=f'No.{i+1} CO$_{2}$ uptake (mol/kg)',
                            filename =dir_name3+'/'+f_prefix3g+f'q4_cycle{i+1}.gif',
                            y_limits=[-0.005,0.2], interval = 100)
        
        acc1.graph_timelapse_y('P',t_frame=1000,
                            label=f'No.{i+1} Pressure',
                            filename =dir_name3+'/'+f_prefix3g+f'P_cycle{i+1}.gif',
                            y_limits=[0.9,7.5], interval = 100)
        acc1.graph_timelapse_y(0,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$ mole frac.',
                            filename =dir_name3+'/'+f_prefix3g+f'y1_cycle{i+1}.gif',
                            y_limits=[-0.04,1.04], interval = 100)
        acc1.graph_timelapse_y(1,t_frame=1000,
                            label=f'No.{i+1} CO mole frac.',
                            filename =dir_name3+'/'+f_prefix3g+f'y2_cycle{i+1}.gif',
                            y_limits=[-0.04,1.04], interval = 100)
        acc1.graph_timelapse_y(2,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$O mole frac.',
                            filename =dir_name3+'/'+f_prefix3g+f'y3_cycle{i+1}.gif',
                            y_limits=[-0.04,1.04], interval = 100)
        acc1.graph_timelapse_y(3,t_frame=1000,
                            label=f'No.{i+1} CO$_{2}$ mole frac.',
                            filename =dir_name3+'/'+f_prefix3g+f'y4_cycle{i+1}.gif',
                            y_limits=[-0.04,1.04], interval = 100)

    TP_fd3 = T_feed3 - (T_feed3 - T_feed2)*np.exp(-t_ran1/tau1)
    TP_feed3 = np.broadcast_to(TP_fd3[:, np.newaxis], (30001, N))

    C_ov_step3 = acc1.y_res[:, 0*N:1*N] + acc1.y_res[:, 1*N:2*N] + acc1.y_res[:, 2*N:3*N] + acc1.y_res[:, 3*N:4*N]
    P_ov_step3_T = C_ov_step3*R_gas*TP_feed3/1E5

    #purge out
    P_out_pur = P_ov_step3_T[:,0]
    u_out_pur = Cv_end_cal3*(P_out_pur-P_end3)
    CO_out_pur = u_out_pur*acc1.y_res[:, 1*N:2*N][:,0]*A_cros*epsi
    inte_CO_out_pur = trapz(CO_out_pur, t_ran1)*2

    #purge all gas out
    all_out_pur = u_out_pur*C_ov_step3[:,0]*A_cros*epsi
    inte_out_pur = trapz(all_out_pur, t_ran1)*2

    H2_out_pur = u_out_pur*acc1.y_res[:, 0*N:1*N][:,0]*A_cros*epsi
    inte_H2_out_pur = trapz(H2_out_pur, t_ran1)*2

    CO_out_pur = u_out_pur*acc1.y_res[:, 1*N:2*N][:,0]*A_cros*epsi
    inte_CO_out_pur = trapz(CO_out_pur, t_ran1)*2

    H2O_out_pur = u_out_pur*acc1.y_res[:, 2*N:3*N][:,0]*A_cros*epsi
    inte_H2O_out_pur = trapz(H2O_out_pur, t_ran1)*2

    CO2_out_pur = u_out_pur*acc1.y_res[:, 3*N:4*N][:,0]*A_cros*epsi
    inte_CO2_out_pur = trapz(CO2_out_pur, t_ran1)*2

    #purge H2 in
    C1_feed_pur = y_feed_list3[0]*P_feed_cal3*1E5/R_gas/T_feed3
    H2_in_pur = u_feed3*C1_feed_pur*A_cros*epsi
    inte_H2_in_pur = trapz(H2_in_pur * np.ones_like(t_ran1), t_ran1)*2


    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # step4 Repressurization
    print(f'This simulation is No.{i+1} Repressurization')
    print(datetime.now())
    before = time.time()
    

    acc1.next_init(Cq_return=True, change_init=True)

    #y_feed_list4 = [0.1,0.3,0.4,0.2]  
    y_feed_list4 = [0,0.3,0.7,0] 
    T_feed4 = 623   # T_feed1 = 623 K (350 dgr C)
    P_feed4 = 6.0
    u_feed4 = 0.01   # u_feed1 = 0.05 m/s

    #acc1.feed_condi_rep(y_feed_arr, P_feed4, T_feed4, )
    #acc1.next_init_p2(T_feed3, T_feed4, Cq_return = False, change_init = True)

    acc1.feed_condi_y(y_feed_list4, P_feed4, T_feed4, u_feed4, 
                    Cv_in= 0.03, const_u = False,)
    # Outlet condition
    P_end4 = 1.0    # P_end1 = 5.5 (bar)
    Cv_out4 = 0.00  # Cv_out1 = 0.02
    T_end4 = 623
    y_end_list4 = [1,0,0,0]
    acc1.outlet_condi_y(y_end_list4, P_end4,
                        T_end4, Cv_out4)

    acc1.Flow_direction(True)
    acc1.run_mamo(t_ran2)

    acc1.next_init(Cq_return= False, change_init= True)

    after_all = time.time()
    after = time.time()
    print(f'It takes {math.floor(after - before)} sec')
    print(f'Cycle No.{i+1} Duration :{math.floor(after_all - before_all)} sec')
    print('')

    figsize_test = [12.5,5]
    t_interval = 2000
    ShowGraph_mol = False
    acc1.graph_still('P', t_frame = t_interval,
                        label = f'No.{i+1} Pressure (bar)',
                        filename=dir_name4+'/'+f_prefix4+f'P_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)

    # Uptake: H2 & CO2
    acc1.graph_still(4, t_frame = t_interval,
                        label = f'No.{i+1} H$_{2}$ uptake (mol/kg)',
                        filename=dir_name4+'/'+f_prefix4+f'q1_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still(7, t_frame = t_interval,
                        label = f'No.{i+1} CO$_{2}$ uptake (mol/kg)',
                        filename=dir_name4+'/'+f_prefix4+f'q4_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    
    # Concentration: H2 & CO2
    acc1.graph_still(0, t_frame = t_interval,
                        label = f'No.{i+1} H$_{2}$ Concentration (mol/m$^{3}$)',
                        filename=dir_name4+'/'+f_prefix4+f'c1_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still(1, t_frame = t_interval,
                        label = f'No.{i+1} CO Concentration (mol/m$^{3}$)',
                        filename=dir_name4+'/'+f_prefix4+f'c2_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still(2, t_frame = t_interval,
                        label = f'No.{i+1} H$_{2}$O Concentration (mol/m$^{3}$)',
                        filename=dir_name4+'/'+f_prefix4+f'c3_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still(3, t_frame = t_interval,
                        label = f'No.{i+1} CO$_{2}$ Concentration (mol/m$^{3}$)',
                        filename=dir_name4+'/'+f_prefix4+f'c4_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)

    # Mole fraction: H2 & CO2
    acc1.graph_still_mol(0, t_frame = t_interval,
                        label = f'No.{i+1} H$_{2}$ mole frac.',
                        filename=dir_name4+'/'+f_prefix4+f'y1_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still_mol(1, t_frame = t_interval,
                        label = f'No.{i+1} CO mole frac.',
                        filename=dir_name4+'/'+f_prefix4+f'y2_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    acc1.graph_still_mol(3, t_frame = t_interval,
                        label = f'No.{i+1} CO$_{2}$ mole frac.',
                        filename=dir_name4+'/'+f_prefix4+f'y4_cycle{i+1}.png',
                        figsize = figsize_test,
                        show = ShowGraph_mol)
    
    #Gif making
    if i+1 in [1, 25, 50]:
        acc1.graph_timelapse(0,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$ Concentration (mol/m$^{3}$)',
                            filename =dir_name4+'/'+f_prefix4g+f'gif_c1_cycle{i+1}.gif',
                            y_limits=[-5,120], interval = 100)
        acc1.graph_timelapse(1,t_frame=1000,
                            label=f'No.{i+1} CO Concentration (mol/m$^{3}$)',
                            filename =dir_name4+'/'+f_prefix4g+f'c2_cycle{i+1}.gif',
                            y_limits=[-5,80], interval = 100)
        acc1.graph_timelapse(2,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$O Concentration (mol/m$^{3}$)',
                            filename =dir_name4+'/'+f_prefix4g+f'c3_cycle{i+1}.gif',
                            y_limits=[-5,80], interval = 100)
        acc1.graph_timelapse(3,t_frame=1000,
                            label=f'No.{i+1} CO$_{2}$ Concentration (mol/m$^{3}$)',
                            filename =dir_name4+'/'+f_prefix4g+f'c4_cycle{i+1}.gif',
                            y_limits=[-5,30], interval = 100)
        
        acc1.graph_timelapse(4,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$ uptake (mol/kg)',
                            filename =dir_name4+'/'+f_prefix4g+f'q1_cycle{i+1}.gif',
                            y_limits=[-0.0005,0.005], interval = 100)
        acc1.graph_timelapse(5,t_frame=1000,
                            label=f'No.{i+1} CO uptake (mol/kg)',
                            filename =dir_name4+'/'+f_prefix4g+f'q2_cycle{i+1}.gif',
                            y_limits=[-0.005,0.2], interval = 100)
        acc1.graph_timelapse(6,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$O uptake (mol/kg)',
                            filename =dir_name4+'/'+f_prefix4g+f'q3_cycle{i+1}.gif',
                            y_limits=[-0.005,0.2], interval = 100)
        acc1.graph_timelapse(7,t_frame=1000,
                            label=f'No.{i+1} CO$_{2}$ uptake (mol/kg)',
                            filename =dir_name4+'/'+f_prefix4g+f'q4_cycle{i+1}.gif',
                            y_limits=[-0.005,0.2], interval = 100)
        
        acc1.graph_timelapse_y('P',t_frame=1000,
                            label=f'No.{i+1} Pressure',
                            filename =dir_name4+'/'+f_prefix4g+f'P_cycle{i+1}.gif',
                            y_limits=[0.9,7.5], interval = 100)
        acc1.graph_timelapse_y(0,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$ mole frac.',
                            filename =dir_name4+'/'+f_prefix4g+f'y1_cycle{i+1}.gif',
                            y_limits=[-0.04,1.04], interval = 100)
        acc1.graph_timelapse_y(1,t_frame=1000,
                            label=f'No.{i+1} CO mole frac.',
                            filename =dir_name4+'/'+f_prefix4g+f'y2_cycle{i+1}.gif',
                            y_limits=[-0.04,1.04], interval = 100)
        acc1.graph_timelapse_y(2,t_frame=1000,
                            label=f'No.{i+1} H$_{2}$O mole frac.',
                            filename =dir_name4+'/'+f_prefix4g+f'y3_cycle{i+1}.gif',
                            y_limits=[-0.04,1.04], interval = 100)
        acc1.graph_timelapse_y(3,t_frame=1000,
                            label=f'No.{i+1} CO$_{2}$ mole frac.',
                            filename =dir_name4+'/'+f_prefix4g+f'y4_cycle{i+1}.gif',
                            y_limits=[-0.04,1.04], interval = 100)

    TP_fd4 = T_feed4 - (T_feed4 - T_feed3)*np.exp(-t_ran2/tau2)
    TP_feed4 = np.broadcast_to(TP_fd4[:, np.newaxis], (20001, N))

    C_ov_step4 = acc1.y_res[:, 0*N:1*N] + acc1.y_res[:, 1*N:2*N] + acc1.y_res[:, 2*N:3*N] + acc1.y_res[:, 3*N:4*N]
    P_ov_step4_T = C_ov_step4*R_gas*TP_feed4/1E5

    #rep in
    Cv_in4= 0.03
    C2_feed_rep = y_feed_list4[1]*P_feed4*1E5/R_gas/T_feed4
    P_init_rep = P_ov_step4_T[:,0]
    u_feed_rep = Cv_in4*(P_feed4-P_init_rep)
    CO_in_rep = u_feed_rep*C2_feed_rep*A_cros*epsi
    inte_CO_in_rep = trapz(CO_in_rep, t_ran2)*2

    CO_in_sum = inte_CO_in_fd + inte_CO_in_rep
    CO_out_sum = inte_CO_out_fd + inte_CO_out_dep + inte_CO_out_pur

    CO_conversion = (1-CO_out_sum/CO_in_sum)*100
    CO_conversion_fd = (1-inte_CO_out_fd/inte_CO_in_fd)*100

    all_gas_out = inte_out_fd + inte_out_dep + inte_out_pur
    all_H2_out = inte_H2_out_fd + inte_H2_out_dep + inte_H2_out_pur
    H2_prod = all_H2_out - inte_H2_in_pur

    H2_purity_all = (H2_prod/all_gas_out)*100
    H2_purity_s_feed = (inte_H2_out_fd/inte_out_fd)*100
    H2_purity_H2OX = (inte_H2_out_fd/(inte_out_fd-inte_H2O_out_fd))*100


    all_out_feed = inte_H2_out_fd + inte_CO_out_fd + inte_H2O_out_fd + inte_CO2_out_fd
    molfrac_H2_fd  = round(inte_H2_out_fd  / all_out_feed * 100, 1)
    molfrac_CO_fd  = round(inte_CO_out_fd  / all_out_feed * 100, 1)
    molfrac_H2O_fd = round(inte_H2O_out_fd / all_out_feed * 100, 1)
    molfrac_CO2_fd = round(inte_CO2_out_fd / all_out_feed * 100, 1)

    all_out_dep = inte_H2_out_dep + inte_CO_out_dep + inte_H2O_out_dep + inte_CO2_out_dep
    molfrac_H2_dep  = round(inte_H2_out_dep  / all_out_dep * 100, 1)
    molfrac_CO_dep  = round(inte_CO_out_dep  / all_out_dep * 100, 1)
    molfrac_H2O_dep = round(inte_H2O_out_dep / all_out_dep * 100, 1)
    molfrac_CO2_dep = round(inte_CO2_out_dep / all_out_dep * 100, 1)

    all_out_pur = inte_H2_out_pur + inte_CO_out_pur + inte_H2O_out_pur + inte_CO2_out_pur
    molfrac_H2_pur  = round(inte_H2_out_pur  / all_out_pur * 100, 1)
    molfrac_CO_pur  = round(inte_CO_out_pur  / all_out_pur * 100, 1)
    molfrac_H2O_pur = round(inte_H2O_out_pur / all_out_pur * 100, 1)
    molfrac_CO2_pur = round(inte_CO2_out_pur / all_out_pur * 100, 1)


    #Calculation
    print(f'This simulation is No.{i+1} Pressure')
    print('Feed', P_ov_step1_T[:,0])
    print('Dep', P_ov_step2_T[:,0])
    print('Purge', P_ov_step3_T[:,0])
    print('Rep', P_ov_step4_T[:,0])
    print('')
    print(f'Cycle No.{i+1} result')
    print(f'CO in feed({i+1}):', inte_CO_in_fd)
    print(f'CO in rep({i+1}):', inte_CO_in_rep)
    print(f'CO out feed({i+1}):', inte_CO_out_fd)
    print(f'CO out dep({i+1}):', inte_CO_out_dep)
    print(f'CO out purge({i+1}):', inte_CO_out_pur)
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    print(f'CO in({i+1}):', CO_in_sum)
    print(f'CO out({i+1}):', CO_out_sum)
    print(f'CO conversion ({i+1}): {CO_conversion:.2f}%')
    print(f'CO conversion feed ({i+1}): {CO_conversion_fd:.2f}%')
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    print('gas out feed', inte_out_fd)
    print('gas out dep', inte_out_dep)
    print('gas out purge', inte_out_pur)
    print('H2 out feed', inte_H2_out_fd)
    print('H2 out dep', inte_H2_out_dep)
    print('H2 out purge', inte_H2_out_pur)
    print('H2 in purge', inte_H2_in_pur)
    print(f'H2 purity (feed) ({i+1}): {H2_purity_s_feed:.2f}%')
    print(f'H2 purity (H2O X) ({i+1}): {H2_purity_H2OX:.2f}%')
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    print("molfrac_H2_fd", molfrac_H2_fd)
    print("molfrac_CO_fd", molfrac_CO_fd)
    print("molfrac_H2O_fd", molfrac_H2O_fd)
    print("molfrac_CO2_fd", molfrac_CO2_fd)
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    print("molfrac_H2_dep", molfrac_H2_dep)
    print("molfrac_CO_dep", molfrac_CO_dep)
    print("molfrac_H2O_dep", molfrac_H2O_dep)
    print("molfrac_CO2_dep", molfrac_CO2_dep)
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    print("molfrac_H2_pur", molfrac_H2_pur)
    print("molfrac_CO_pur", molfrac_CO_pur)
    print("molfrac_H2O_pur", molfrac_H2O_pur)
    print("molfrac_CO2_pur", molfrac_CO2_pur)
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    print("mole_H2_fd", inte_H2_out_fd)
    print("mole_CO_fd",inte_CO_out_fd)
    print("mole_H2O_fd", inte_H2O_out_fd)
    print("mole_CO2_fd", inte_CO2_out_fd)
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    print("mole_H2_dep", inte_H2_out_dep)
    print("mole_CO_dep",inte_CO_out_dep)
    print("mole_H2O_dep", inte_H2O_out_dep)
    print("mole_CO2_dep", inte_CO2_out_dep)
    print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    print("mole_H2_pur", inte_H2_out_pur)
    print("mole_CO_pur",inte_CO_out_pur)
    print("mole_H2O_pur", inte_H2O_out_pur)
    print("mole_CO2_pur", inte_CO2_out_pur)
    print('')
    print('')

    CO_conversion_list.append(CO_conversion)
    CO_conversion_list_feed.append(CO_conversion_fd)
    H2_purity_feed_list.append(H2_purity_s_feed)
    H2_purity_H2OX_list.append(H2_purity_H2OX)

    molfrac_H2_fd_list.append(molfrac_H2_fd)
    molfrac_CO_fd_list.append(molfrac_CO_fd)
    molfrac_H2O_fd_list.append(molfrac_H2O_fd)
    molfrac_CO2_fd_list.append(molfrac_CO2_fd)

    molfrac_H2_dep_list.append(molfrac_H2_dep)
    molfrac_CO_dep_list.append(molfrac_CO_dep)
    molfrac_H2O_dep_list.append(molfrac_H2O_dep)
    molfrac_CO2_dep_list.append(molfrac_CO2_dep)

    molfrac_H2_pur_list.append(molfrac_H2_pur)
    molfrac_CO_pur_list.append(molfrac_CO_pur)
    molfrac_H2O_pur_list.append(molfrac_H2O_pur)
    molfrac_CO2_pur_list.append(molfrac_CO2_pur)




end_time = time.time()  
total_duration = math.floor(end_time - start_time)  

print(f'Total duration : {total_duration} sec')
print('')


def plot_cycle_graph(x, y, ylabel, title, filename, color='k', label=None, figsize=(10,5), dpi=150):
    """
    사이클 기반 그래프를 저장하는 범용 함수
    
    Parameters:
    - x: x축 데이터 (예: cycle number)
    - y: y축 데이터 (예: CO conversion list)
    - ylabel: y축 라벨
    - title: 그래프 제목
    - filename: 저장할 파일 이름 (예: 'CO_conversion.png')
    - color: 선 색상 (기본: 검은색)
    - label: 범례 라벨
    - figsize: 그림 크기
    - dpi: 이미지 저장 품질
    """
    plt.figure(figsize=figsize)
    plt.plot(x, y, marker='o', linestyle='-', color=color, label=label if label else title)
    plt.xlabel("Cycle Number")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()  # 메모리 절약을 위해 close() 필수


x_data = range(1, cycle_N + 1)

# CO conversion 전체
plot_cycle_graph(x_data, CO_conversion_list,
                 ylabel="CO Conversion (%)",
                 title="CO Conversion vs. Cycle Number",
                 filename="CO_conversion.png",
                 label="CO Conversion")

# CO conversion (Feed only)
plot_cycle_graph(x_data, CO_conversion_list_feed,
                 ylabel="CO Conversion (%)",
                 title="CO Conversion (Feed Only) vs. Cycle Number",
                 filename="CO_conversion_feed.png",
                 label="CO Conversion (Feed)")

# H2 purity (Feed)
plot_cycle_graph(x_data, H2_purity_feed_list,
                 ylabel="H2 purity (%)",
                 title="H2 Purity vs. Cycle Number",
                 filename="H2_purity.png",
                 label="H2 Purity")

# H2 purity (H2O X)
plot_cycle_graph(x_data, H2_purity_H2OX_list,
                 ylabel="H2 purity (%)",
                 title="H2 Purity vs. Cycle Number",
                 filename="H2_purity_H2O(X).png",
                 label="H2 Purity")


# mole frac feed
plot_cycle_graph(x_data, molfrac_H2_fd_list,
                 ylabel="H2 mol frac. (%)",
                 title="H2 mol frac. vs. Cycle Number",
                 filename="molfrac_H2_fd.png",
                 label="H2 mol frac.")

plot_cycle_graph(x_data, molfrac_CO_fd_list,
                 ylabel="CO mol frac. (%)",
                 title="CO mol frac. vs. Cycle Number",
                 filename="molfrac_CO_fd.png",
                 label="CO mol frac.")

plot_cycle_graph(x_data, molfrac_H2O_fd_list,
                 ylabel="H2O mol frac. (%)",
                 title="H2O mol frac. vs. Cycle Number",
                 filename="molfrac_H2O_fd.png",
                 label="H2O mol frac.")

plot_cycle_graph(x_data, molfrac_CO2_fd_list,
                 ylabel="CO2 mol frac. (%)",
                 title="CO2 mol frac. vs. Cycle Number",
                 filename="molfrac_CO2_fd.png",
                 label="CO2 mol frac.")

# mole frac dep
plot_cycle_graph(x_data, molfrac_H2_dep_list,
                 ylabel="H2 mol frac. (%)",
                 title="H2 mol frac. vs. Cycle Number",
                 filename="molfrac_H2_dep.png",
                 label="H2 mol frac.")

plot_cycle_graph(x_data, molfrac_CO_dep_list,
                 ylabel="CO mol frac. (%)",
                 title="CO mol frac. vs. Cycle Number",
                 filename="molfrac_CO_dep.png",
                 label="CO mol frac.")

plot_cycle_graph(x_data, molfrac_H2O_dep_list,
                 ylabel="H2O mol frac. (%)",
                 title="H2O mol frac. vs. Cycle Number",
                 filename="molfrac_H2O_dep.png",
                 label="H2O mol frac.")

plot_cycle_graph(x_data, molfrac_CO2_dep_list,
                 ylabel="CO2 mol frac. (%)",
                 title="CO2 mol frac. vs. Cycle Number",
                 filename="molfrac_CO2_dep.png",
                 label="CO2 mol frac.")

# mole frac pur
plot_cycle_graph(x_data, molfrac_H2_pur_list,
                 ylabel="H2 mol frac. (%)",
                 title="H2 mol frac. vs. Cycle Number",
                 filename="molfrac_H2_pur.png",
                 label="H2 mol frac.")

plot_cycle_graph(x_data, molfrac_CO_pur_list,
                 ylabel="CO mol frac. (%)",
                 title="CO mol frac. vs. Cycle Number",
                 filename="molfrac_CO_pur.png",
                 label="CO mol frac.")

plot_cycle_graph(x_data, molfrac_H2O_pur_list,
                 ylabel="H2O mol frac. (%)",
                 title="H2O mol frac. vs. Cycle Number",
                 filename="molfrac_H2O_pur.png",
                 label="H2O mol frac.")

plot_cycle_graph(x_data, molfrac_CO2_pur_list,
                 ylabel="CO2 mol frac. (%)",
                 title="CO2 mol frac. vs. Cycle Number",
                 filename="molfrac_CO2_pur.png",
                 label="CO2 mol frac.")














   # %%

#CO conversion graph
plt.figure(figsize=(10,5))
plt.plot(range(1, cycle_N + 1), CO_conversion_list, marker='o', linestyle='-', color='k', label="CO Conversion")
plt.xlabel("Cycle Number")
plt.ylabel("CO Conversion (%)")
plt.title("CO Conversion vs. Cycle Number")
plt.legend()
plt.grid()
plt.savefig('CO_conversion.png', dpi = 150, bbox_inches = 'tight')

#CO conversion graph feed only
plt.figure(figsize=(10,5))
plt.plot(range(1, cycle_N + 1), CO_conversion_list_feed, marker='o', linestyle='-', color='k', label="CO Conversion")
plt.xlabel("Cycle Number")
plt.ylabel("CO Conversion (%)")
plt.title("CO Conversion vs. Cycle Number")
plt.legend()
plt.grid()
plt.savefig('CO_conversion_feed.png', dpi = 150, bbox_inches = 'tight')

#H2_purity feed
plt.figure(figsize=(10,5))
plt.plot(range(1, cycle_N + 1), H2_purity_feed_list, marker='o', linestyle='-', color='k', label="H2 purity")
plt.xlabel("Cycle Number")
plt.ylabel("H2 purity (%)")
plt.title("H2 purity vs. Cycle Number")
plt.legend()
plt.grid()
plt.savefig('H2_ purity.png', dpi = 150, bbox_inches = 'tight')


# %%
