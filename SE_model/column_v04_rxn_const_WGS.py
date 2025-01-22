
# %%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
# %%
# Reaction parameters
# %%
T_g  = 773 # K # Temperature (400 oC)
T_init = 773
K_eq = 10**(-2.4198 + 0.0003855*T_g + 2180.6/T_g) # Equilibrium constant
k_r1 = 0.003 # (mol/m^3/s/bar^2) r1 = k_r1*P2*P3
k_r2 = k_r1/K_eq # (mol/m^3/s/bar^2) r2 = k_r2*P1*P4

# %%
# Spatial setting for z-axis
# %%
# Number of spatial node
N = 41
L = 2.0
z_dom = np.linspace(0,L,N)

# Discretization matrix d & dd

# d = ?
h_arr = (z_dom[1] - z_dom[0])*np.ones([N,])
d_arr = 1/h_arr

d_mid = np.diag(d_arr, 0)
d_low = np.diag(d_arr[:-1],-1)
d = d_mid - d_low
d[0,:] = 0

# dd = ?
dd_arr = 1/h_arr**2
dd_upp = np.diag(dd_arr[:-1],1)
dd_mid = np.diag(dd_arr,0)
dd_low = np.diag(dd_arr[:-1], -1)

dd = dd_upp - 2*dd_mid + dd_low
d[0,:] = 0
d[-1,:]=0

# %% 
# Ergun equation
# %%
# Ergun equation function
def Ergun(P,z,d_p, mu, rho_g,void_frac):
    dPdz = (P[1:]-P[:-1])/(z[1:]-z[:-1])
    Aterm = 1.75*(1-void_frac)/void_frac*rho_g/d_p
    Bterm = 150*mu*(1-void_frac)**2/void_frac**2/d_p**2
    Cterm = dPdz
    arg_u_pos = dPdz <= 0
    arg_u_neg = arg_u_pos < 0.5
    u_return = np.zeros_like(dPdz)
    # u > 0 //due to dPdz < 0  ||  Cterm < 0  
    Apos = Aterm[arg_u_pos]
    Bpos = Bterm
    Cpos = Cterm[arg_u_pos]
    u_return[arg_u_pos] = (-Bpos + np.sqrt(Bpos**2 - 4*Apos*Cpos))/2/Apos
    # u < 0 //due to dPdz > 0  ||  Cterm > 0  
    Aneg = Aterm[arg_u_neg]
    Bneg = Bterm
    Cneg = Cterm[arg_u_neg]
    u_return[arg_u_neg] = (-Bneg + np.sqrt(Bneg**2 + 4*Aneg*Cneg))/2/(-Aneg)

    return u_return, arg_u_pos, arg_u_neg
#plt.plot(z_dom, P_ov_test)
# Ergun for binary system
def Ergun_qu(C_list, T_gas, z,
              d_p, mu, Mw_list, void_frac):
    C1, C2, C3, C4 = C_list
    Mw1,Mw2,Mw3,Mw4, = Mw_list
    C1_mid = (C1[1:] + C1[:-1])/2
    C2_mid = (C2[1:] + C2[:-1])/2
    C3_mid = (C3[1:] + C3[:-1])/2
    C4_mid = (C4[1:] + C4[:-1])/2
    rho_g_tmp = C1_mid*Mw1 + C2_mid*Mw2 + C3_mid*Mw3+ C4_mid*Mw4
    C_ov_tmp = C1 + C2 + C3 + C4
    P_ov_tmp = C_ov_tmp*R_gas*T_gas
    u_ret, arg_u_pos, arg_u_neg = Ergun(P_ov_tmp, 
                                        z, d_p, mu, rho_g_tmp, void_frac)
    return u_ret, arg_u_pos, arg_u_neg


# %%
# Isotherm Model 
# %%
qm1 = 0.02 # H2
qm2 = 0.3 # CO
qm3 = 0.6 # H2O
qm4 = 3.0 # CO2

b1 = 0.05
b2 = 0.1
b3 = 0.3
b4 = 0.5

def iso(P1,P2,P3,P4):
    deno = 1+b1*P1 + b2*P2 + b3*P3 + b4*P4
    nume1 = qm1*b1*P1
    nume2 = qm2*b2*P2
    nume3 = qm3*b3*P3
    nume4 = qm4*b4*P4

    q1 = nume1/deno
    q2 = nume2/deno
    q3 = nume3/deno
    q4 = nume4/deno
    return q1, q2, q3, q4

# %%
# Initial Values
# %%
# Initial P, T, C1, C2, q1, q2, y0
P_init = 3.2E5*np.ones([N,])    # (Pa)
#T_init = 773
R_gas = 8.3145
# Gas phase
C1_init = 1.0*P_init/R_gas/T_init*np.ones([N,]) # H2
C2_init = 0.0*P_init/R_gas/T_init*np.ones([N,]) # CO
C3_init = 0.0*P_init/R_gas/T_init*np.ones([N,]) # H2O
C4_init = 0.0*P_init/R_gas/T_init*np.ones([N,]) # CO2
#Cov_init = C1_init + C2_init + C3_init + C4_init
P1_init = C1_init*R_gas*T_init
P2_init = C2_init*R_gas*T_init
P3_init = C3_init*R_gas*T_init
P4_init = C4_init*R_gas*T_init
 
# Solid phase
q1_init, q2_init, q3_init, q4_init = iso(P1_init*1E-5, P2_init*1E-5,
                                         P3_init*1E-5, P4_init*1E-5,)
y0 = np.concatenate([C1_init, C2_init, C3_init, C4_init,
                    q1_init, q2_init, q3_init, q4_init])
# %%
# Boundary Conditions
# %%
# Pressure & velcotiy conditions
P_end = 3.0       # (bar)
Cv_out = 2E-2 
u_feed = 0.05
P_feed = 4.5    # (bar)

# Inlet conditions
T_feed = T_init
C1_feed = 0.0*P_feed*1E5/R_gas/T_feed # H2  (prod)
C2_feed = 0.5*P_feed*1E5/R_gas/T_feed # CO  (reac)
C3_feed = 0.5*P_feed*1E5/R_gas/T_feed # H2O (reac)
C4_feed = 0.0*P_feed*1E5/R_gas/T_feed # CO2 (prod)
# %%
# Model for odeint
# %%
# Additional parameters
D_AB = 1E-8
h = h_arr[0]
#T_g = T_init
#u_g = 0.01      # (m/s) : advective velocity

rho_s = 1000    # (kg/m^3)
rho_cat = 1000    # (kg/m^3)
epsi = 0.4     # voide fraction of pellet
x_cat= 0.7      # catalyst fraction in pellet (cat/(cat+ads))

#k1, k2 = [0.2,0.2]
k1, k2, k3, k4 = [0.02,]*4
d_particle = 2E-3   # (m) : pellet particle size
mu_gas = 1.8E-5 # (Pa s) : viscosity of gas (air)  
Mw_gas = np.array([2, 28, 18, 44])*1E-3 # (kg/mol) : molar weight for each component

# %%
# Reaction kinetics
# %%
#K_eq = 10**(-2.4198 + 0.0003855*T_g + 2180.6/T_g) # Equilibrium constant
#k_r1 = 0.02 # (mol/m^3/s/bar^2) r1 = k_r1*P2*P3
#k_r2 = k_r1/K_eq # (mol/m^3/s/bar^2) r2 = k_r2*P1*P4

# Backflow at z = L 
C1_L = 1.0*P_end*1E5/R_gas/T_g  # H2 (prod)
C2_L = 0.0*P_end*1E5/R_gas/T_g # CO (reac)
C3_L = 0.0*P_end*1E5/R_gas/T_g # H2O (reac)
C4_L = 0.0*P_end*1E5/R_gas/T_g    # CO2 (prod)

# %%
# PDE -> ODE model
# %%
# PDE -> ODE model
def model_col(y,t ):
    C1 = y[:N]
    C2 = y[1*N : 2*N]
    C3 = y[2*N : 3*N]
    C4 = y[3*N : 4*N]
    
    q1 = y[4*N : 5*N]
    q2 = y[5*N : 6*N]
    q3 = y[6*N : 7*N]
    q4 = y[7*N : 8*N]

    # Pressure
    P1 = C1*R_gas*T_g/1E5   # in bar
    P2 = C2*R_gas*T_g/1E5   # in bar
    P3 = C3*R_gas*T_g/1E5   # in bar
    P4 = C4*R_gas*T_g/1E5   # in bar
    
    # velocity from Ergun equation
    u_g, arg_u_posi, arg_u_nega = Ergun_qu([C1,C2,C3,C4], T_g, z_dom,
                                           d_particle, mu_gas,
                                           Mw_gas, epsi)

    # Valve equation
    P_ov_end = P1[-1]+P2[-1] + P3[-1]+ P4[-1]
    u_out = Cv_out*(P_ov_end - P_end) # (bar) -> m/s

    # uC1
    uC1_tmp = u_g*C1[:-1]
    uC1_back = u_g*C1[1:]
    uC1_tmp[arg_u_nega] = uC1_back[arg_u_nega]
    uC1_z0 = C1_feed*np.max([u_feed, 0]) + C1[0]*np.min([u_feed, 0])
    uC1_zL = C1[-1]*np.max([u_out,0]) + C1_L*np.min([u_out, 0])
    uC1 = np.concatenate([ [uC1_z0,], uC1_tmp, [uC1_zL,], ] )
    duC1dz = (uC1[1:]-uC1[:-1])/h
    # uC2
    uC2_tmp = u_g*C2[:-1]
    uC2_back = u_g*C2[1:]
    uC2_tmp[arg_u_nega] = uC2_back[arg_u_nega]
    uC2_z0 = C2_feed*np.max([u_feed,0]) + C2[0]*np.min([u_feed, 0])
    uC2_zL = C2[-1]*np.max([u_out,0]) + C2_L*np.min([u_out, 0])
    uC2 = np.concatenate([[uC2_z0], uC2_tmp, [uC2_zL],])
    duC2dz = (uC2[1:]-uC2[:-1])/h
    # uC3
    uC3_tmp = u_g*C3[:-1]
    uC3_back = u_g*C3[1:]
    uC3_tmp[arg_u_nega] = uC3_back[arg_u_nega]
    uC3_z0 = C3_feed*np.max([u_feed,0]) + C3[0]*np.min([u_feed, 0])
    uC3_zL = C3[-1]*np.max([u_out,0]) + C3_L*np.min([u_out, 0])
    uC3 = np.concatenate([[uC3_z0], uC3_tmp, [uC3_zL],])
    duC3dz = (uC3[1:]-uC3[:-1])/h
    # uC4
    uC4_tmp = u_g*C4[:-1]
    uC4_back = u_g*C4[1:]
    uC4_tmp[arg_u_nega] = uC4_back[arg_u_nega]
    uC4_z0 = C4_feed*np.max([u_feed,0]) + C4[0]*np.min([u_feed, 0])
    uC4_zL = C4[-1]*np.max([u_out,0]) + C4_L*np.min([u_out, 0])
    uC4 = np.concatenate([[uC4_z0], uC4_tmp, [uC4_zL],])
    duC4dz = (uC4[1:]-uC4[:-1])/h

    # Isotherm
    q1sta, q2sta, q3sta, q4sta = iso(P1,P2,P3,P4,)

    # Reactions
    r1 = k_r1*P2*P3
    r2 = k_r2*P1*P4

    # Discretization
    ddC1 = dd@C1
    ddC2 = dd@C2
    ddC3 = dd@C3
    ddC4 = dd@C4

    # LDF
    dq1dt = k1*(q1sta - q1)
    dq2dt = k2*(q2sta - q2)
    dq3dt = k3*(q3sta - q3)
    dq4dt = k4*(q4sta - q4)

    # Mass balanace
    m_ad = rho_s*(1-epsi)/epsi*(1-x_cat)
    m_cat = rho_cat*(1-epsi)/epsi*x_cat
    dC1dt = D_AB*ddC1 - duC1dz - m_ad*dq1dt + m_cat*(r1-r2) #H2
    dC2dt = D_AB*ddC2 - duC2dz - m_ad*dq2dt + m_cat*(-r1+r2) #CO
    dC3dt = D_AB*ddC3 - duC3dz - m_ad*dq3dt + m_cat*(-r1+r2) #H2O
    dC4dt = D_AB*ddC4 - duC4dz - m_ad*dq4dt + m_cat*(r1-r2) #CO2

    # Boundary conditions
    dC1dt[0] = - duC1dz[0] - m_ad*dq1dt[0] + m_cat*(+r1[0]-r2[0])
    dC2dt[0] = - duC2dz[0] - m_ad*dq2dt[0] + m_cat*(-r1[0]+r2[0])
    dC3dt[0] = - duC3dz[0] - m_ad*dq3dt[0] + m_cat*(-r1[0]+r2[0])
    dC4dt[0] = - duC4dz[0] - m_ad*dq4dt[0] + m_cat*(+r1[0]-r2[0])

    dC1dt[-1] = - duC1dz[-1] - m_ad*dq1dt[-1] + m_cat*(+r1[-1]-r2[-1])
    dC2dt[-1] = - duC2dz[-1] - m_ad*dq2dt[-1] + m_cat*(-r1[-1]+r2[-1])
    dC3dt[-1] = - duC3dz[-1] - m_ad*dq3dt[-1] + m_cat*(-r1[-1]+r2[-1])
    dC4dt[-1] = - duC4dz[-1] - m_ad*dq4dt[-1] + m_cat*(+r1[-1]-r2[-1])
    
    dydt = np.concatenate([dC1dt,dC2dt, dC3dt, dC4dt, dq1dt, dq2dt, dq3dt,dq4dt])
    return dydt

# %%
# Run 
# %%
t_ran = np.arange(0,320+0.0025,0.0025)
y_res = odeint(model_col, y0, t_ran )
print(y_res.shape)
# %%
# Function for Plotting Graphs
# %%

def graph_t(y_targ, z_domain, t_span, index, label, filename, bbox_pos=[1.42, 0.92]):
    plt.figure()
    cc = 0
    ls_list = ['-','--','-.',':']
    for ii, tt in zip(index, t_span):
        C_samp = y_targ[ii,:]
        plt.plot(z_domain, C_samp, 'k',
                 linestyle = ls_list[cc%len(ls_list)],
                 label = f't = {tt} sec'
                 )
        cc += 1
    plt.legend(fontsize = 13, fancybox = True,
               shadow = True, ncol = 2,
               loc = 'upper center',
               bbox_to_anchor = bbox_pos,)
    plt.xlabel('z-axis (m)',
                fontsize = 13)
    plt.ylabel(label, fontsize = 13)
    plt.grid(ls = '--')
    plt.savefig(filename, dpi = 150, bbox_inches = 'tight')
# %%
# Determine time points to sample 
# %%
t_sample = t_ran[::8000]
ii_arr = np.arange(len(t_ran))[::8000]

# %%
# Graph drawing for P (pressure)
# %%
# Pressure
C_ov = y_res[:, 0*N:1*N]+y_res[:, 1*N:2*N]+y_res[:, 2*N:3*N]+y_res[:, 3*N:4*N]
P_ov = C_ov*R_gas*T_g/1E5
graph_t(P_ov, z_dom, t_sample, ii_arr, 
        'Pressure (bar)', 'res_P_profile.png', bbox_pos = (1.42, 0.92))

# %%
# Graph drawing for C (concentration)
# %%
# C1 Profile (H2)
graph_t(y_res[:,0*N:1*N], z_dom, t_sample, ii_arr,
        'H$_{2}$ Concentration (mol/m$^{3}$)', 'res_C1_profile_H2.png')

# C4 Profile (CO2)
graph_t(y_res[:,3*N:4*N], z_dom, t_sample, ii_arr,
        'CO$_{2}$ Concentration (mol/m$^{3}$)', 'res_C4_profile_CO2.png')

# %%
# Grpah drawing for q (uptake)
# %%
# q1 Profile (H2)
graph_t(y_res[:,4*N:5*N], z_dom, t_sample, ii_arr,
        'H$_{2}$ uptake (mol/kg)', 'res_q1_profile_H2.png')
# q4 profile (CO2)
graph_t(y_res[:,7*N:8*N], z_dom, t_sample, ii_arr,
        'CO$_{2}$ uptake (mol/kg)', 'res_q4_profile_CO2.png')

# %%
# Graph drawing for y (mole fraction)
# %%
# y1 Profile (H2 mole fraction)
C_ov = y_res[:, 0*N:1*N]+y_res[:, 1*N:2*N]+y_res[:, 2*N:3*N]+y_res[:, 3*N:4*N]

graph_t(y_res[:,0*N:1*N]/C_ov, z_dom, t_sample, ii_arr, 
        'H$_{2}$ moe frac. (mol/mol)', 'res_y1_profile.png', 
        bbox_pos=(1.42, 0.92))
# y2 Profile (CO mole fraction)
graph_t(y_res[:,1*N:2*N]/C_ov, z_dom, t_sample, ii_arr,
        'CO mole frac. (mol/mol)', 'res_y2_profile_CO.png', 
        bbox_pos = (1.42, 0.92))

# y3 Profile (H2O mole fraction)
graph_t(y_res[:,2*N:3*N]/C_ov, z_dom, t_sample, ii_arr,
        'H$_{2}$O mole frac. (mol/mol)', 'res_y3_profile_H2O.png',
        bbox_pos = (1.42, 0.92))

# y4 Profile (CO2 mole fraction)
graph_t(y_res[:,3*N:4*N]/C_ov, z_dom, t_sample, ii_arr,
        'CO$_{2}$ mole frac. (mol/mol)', 'res_y4_profile_CO2.png', 
        bbox_pos = (1.42, 0.92))
# %%
# =-=-=-=-=-==-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=

import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
import os
def create_gif_fixed_yaxis(y_targ, z_domain, t_span, index, label, filename, 
                           bbox_pos=[1.42, 0.92], y_limits=None, interval=100):
    """
    Create a GIF showing the profile of a variable over time with a fixed y-axis.

    Parameters:
    - y_targ: 2D array, data to plot (rows correspond to time, columns to spatial domain).
    - z_domain: 1D array, spatial domain (x-axis).
    - t_span: list or array, time points to annotate.
    - index: list or array, indices of time points to include in the GIF.
    - label: str, y-axis label.
    - filename: str, name of the output GIF file.
    - bbox_pos: list or tuple, position of the legend box (default: [1.42, 0.92]).
    - y_limits: tuple or list, fixed y-axis limits as (y_min, y_max) (default: None).
    """
    fig, ax = plt.subplots()

    # Define the line plot
    line, = ax.plot([], [], 'k-', lw=2)
    ax.set_xlabel('z-axis (m)', fontsize=13)
    ax.set_ylabel(label, fontsize=13)
    ax.grid(ls='--')
    ax.legend(fontsize=13, fancybox=True, shadow=True, ncol=2, loc='upper center', 
              bbox_to_anchor=bbox_pos)
    ax.set_xlim([(2*z_domain[0]-z_domain[1]), (2*z_domain[-1]-z_domain[-2])])

    # Set fixed y-axis limits if provided
    if y_limits:
        ax.set_ylim(y_limits)

    # Update function for each frame in the GIF
    def update(frame_idx):
        tt = t_span[frame_idx]
        C_samp = y_targ[index[frame_idx], :]
        line.set_data(z_domain, C_samp)
        ax.set_title(f'Profile at t = {tt:.2f} sec', fontsize=14)
        return line,

    # Initialize plot settings
    def init():
        line.set_data([], [])
        return line,

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(index), init_func=init, 
        blit=True, interval = interval )
    
    # Save the GIF
    ani.save(filename, writer='Pillow', fps=1000/interval)
    plt.close(fig)


# %%
# Change the time points to sample
# %%
t_sample = t_ran[::2000]
ii_arr = np.arange(len(t_ran))[::2000]
# %%
# Create GIFs for the pressure, concentration, uptake, and mole fraction profiles
# %%
# Pressure
os.makedirs('results_gif', exist_ok=True)
create_gif_fixed_yaxis(P_ov, z_dom, t_sample, ii_arr,
                       'Pressure (bar)', 'results_gif/res_P_profile.gif', bbox_pos=(1.42, 0.92),
                       y_limits=[-0.01, 5], interval = 100) # Pressure

# %%
# Concentration
create_gif_fixed_yaxis(y_res[:, 0*N:1*N], z_dom, t_sample, ii_arr,
                       'H$_{2}$ Concentration (mol/m$^{3}$)', 
                       'results_gif/res_C1_profile_H2.gif',
                       y_limits = [0, 70],interval = 100) # H2 

create_gif_fixed_yaxis(y_res[:, 3*N:4*N], z_dom, t_sample, ii_arr,
                       'CO$_{2}$ Concentration (mol/m$^{3}$)', 
                       'results_gif/res_C4_profile_CO2.gif',
                       y_limits = [0, 70],interval = 100) # CO2 

# %%
# Uptake
create_gif_fixed_yaxis(y_res[:, 4*N:5*N], z_dom, t_sample, ii_arr,
                       'CO$_{2}$ Concentration (mol/m$^{3}$)', 
                       'results_gif/res_q1_profile_H2.gif',
                       y_limits = [-0.0001, 0.005],interval = 100) # H2
create_gif_fixed_yaxis(y_res[:, 5*N:6*N], z_dom, t_sample, ii_arr,
                       'CO Concentration (mol/m$^{3}$)',
                       'results_gif/res_q2_profile_CO.gif',
                       y_limits = [-0.001, 0.03],interval = 100) # CO

create_gif_fixed_yaxis(y_res[:, 6*N:7*N], z_dom, t_sample, ii_arr,
                       'CO$_{2}$ Concentration (mol/m$^{3}$)', 
                       'results_gif/res_q3_profile_H2O.gif',
                       y_limits = [-0.002, 0.2],interval = 100) # H2O

create_gif_fixed_yaxis(y_res[:, 7*N:8*N], z_dom, t_sample, ii_arr,
                       'CO$_{2}$ Concentration (mol/m$^{3}$)', 
                       'results_gif/res_q4_profile_CO2.gif',
                       y_limits = [-0.03, 1.2],interval = 100) # CO2

# %%
# Mole fraction
create_gif_fixed_yaxis(y_res[:, 0*N:1*N]/C_ov, z_dom, t_sample, ii_arr,
                       'H$_{2}$ mole frac. (mol/mol)',
                       'results_gif/res_y1_profile_H2.gif',
                       y_limits=[-0.01, 1.05], interval=100) # H2
create_gif_fixed_yaxis(y_res[:, 1*N:2*N]/C_ov, z_dom, t_sample, ii_arr,
                       'CO mole frac. (mol/mol)',
                       'results_gif/res_y2_profile_CO.gif',
                       y_limits=[-0.01, 1.05], interval=100) # CO
create_gif_fixed_yaxis(y_res[:, 2*N:3*N]/C_ov, z_dom, t_sample, ii_arr,
                       'H$_{2}$O mole frac. (mol/mol)',
                       'results_gif/res_y3_profile_H2O.gif',
                       y_limits=[-0.01, 1.05], interval=100) # H2O
create_gif_fixed_yaxis(y_res[:, 3*N:4*N]/C_ov, z_dom, t_sample, ii_arr,
                       'CO$_{2}$ mole frac. (mol/mol)',
                       'results_gif/res_y4_profile_CO2.gif',
                       y_limits=[-0.01, 1.05], interval=100) # CO2


# %%
from scipy.integrate import simps

u_end = Cv_out*(P_ov[:,-1] - P_end)
D_column = 0.3
A_area = np.pi*(D_column/2)**2
F_feed_CO = u_feed*C2_feed*epsi*A_area
F_end_CO = u_end*y_res[:,2*N]*epsi*A_area

# number of mole of CO reacted
#N_until = -64000-16000-16000-8000
N_until = -1
n_out_CO = simps(F_end_CO[1:N_until], t_ran[1:N_until])
n_ads_CO = simps(y_res[-1, 5*N:6*N], z_dom)*rho_s*(1-epsi)*A_area

X_CO = (F_feed_CO*t_ran[N_until] - n_out_CO-n_ads_CO)/F_feed_CO/t_ran[N_until]
print('Conversion of CO:\n', X_CO*100)
print('Until', t_ran[N_until], 'sec')

#print(X_CO*100)
#t_ran
plt.figure()
plt.plot(t_ran, F_end_CO, 'k', label = 'CO')
#n_init_H2 = epsi*L*A_area*C1_init

# %%

# %%

        
# LEGACY CODES

'''
# Pressure Profile
C_ov = y_res[:, 0*N:1*N]+y_res[:, 1*N:2*N]+y_res[:, 2*N:3*N]+y_res[:, 3*N:4*N]
# Pressure
plt.figure()
ls_list = ['-','--','-.',':']
cc = 0
for ii, tt in zip(ii_arr, t_sample):
    C_samp = C_ov[ii,:]*R_gas*T_g/1E5
    #C_samp = y_res[ii,7*N:8*N]
    plt.plot(z_dom, C_samp, 'k',
             linestyle = ls_list[cc%len(ls_list)],
             label = f't = {tt}'
             )
    cc += 1
plt.legend(fontsize = 13, fancybox = True,
           shadow = True, ncol = 2,
           loc = 'upper center', 
           bbox_to_anchor = (1.32, 1.17))
plt.xlabel('z-axis (m)', 
            fontsize = 13)
plt.ylabel('Pressure (bar)',
            fontsize = 13)
plt.grid(ls = '--')
plt.savefig('res_P_profile.png', dpi = 150, bbox_inches = 'tight')


plt.figure()
cc = 0
for ii, tt in zip(ii_arr, t_sample):
    #C_samp = C_ov[ii,:]*R_gas*T_g/1E5
    C_samp = y_res[ii,3*N:4*N]
    plt.plot(z_dom, C_samp, 'k',
             linestyle = ls_list[cc%len(ls_list)],
             label = f't = {tt}'
             )
    cc += 1
plt.legend(fontsize = 13, fancybox = True,
           shadow = True, ncol = 2,
           loc = 'upper center', 
           bbox_to_anchor = (1.32, 1.17))
plt.xlabel('z-axis (m)', 
            fontsize = 13)
plt.ylabel('CO$_{2}$ Concentration (mol/m$^{3}$)',
            fontsize = 13)
plt.grid(ls = '--')
plt.savefig('res_C4_profile_CO2.png', dpi = 150, bbox_inches = 'tight')

# C1 Profile H2
plt.figure()
cc = 0
for ii, tt in zip(ii_arr, t_sample):
    #C_samp = C_ov[ii,:]*R_gas*T_g/1E5
    C_samp = y_res[ii,0*N:1*N]
    plt.plot(z_dom, C_samp, 'k',
             linestyle = ls_list[cc%len(ls_list)],
             label = f't = {tt}'
             )
    cc += 1
plt.legend(fontsize = 13, fancybox = True,
           shadow = True, ncol = 2,
           loc = 'upper center', 
           bbox_to_anchor = (1.32, 1.17))
plt.xlabel('z-axis (m)', 
            fontsize = 13)
plt.ylabel('H$_{2}$ Concentration (mol/m$^{3}$)',
            fontsize = 13)
plt.grid(ls = '--')
plt.savefig('res_C1_profile_H2.png', dpi = 150, bbox_inches = 'tight')


# %%
# q Profile
# %%

# q4 Profile
plt.figure()
cc = 0
for ii, tt in zip(ii_arr, t_sample):
    #C_samp = C_ov[ii,:]*R_gas*T_g/1E5
    C_samp = y_res[ii,7*N:8*N]
    plt.plot(z_dom, C_samp, 'k',
             linestyle = ls_list[cc%len(ls_list)],
             label = f't = {tt}'
             )
    cc += 1
plt.legend(fontsize = 13, fancybox = True,
           shadow = True, ncol = 2,
           loc = 'upper center', 
           bbox_to_anchor = (1.32, 1.17))
plt.xlabel('z-axis (m)', 
            fontsize = 13)
plt.ylabel('CO$_{2}$ uptake (mol/kg)',
            fontsize = 13)
plt.grid(ls = '--')
plt.savefig('res_q4_profile_CO2.png', dpi = 150, bbox_inches = 'tight')

# q1 Profile
plt.figure()
cc = 0
for ii, tt in zip(ii_arr, t_sample):
    #C_samp = C_ov[ii,:]*R_gas*T_g/1E5
    C_samp = y_res[ii,4*N:5*N]
    plt.plot(z_dom, C_samp, 'k',
             linestyle = ls_list[cc%len(ls_list)],
             label = f't = {tt}'
             )
    cc += 1
plt.legend(fontsize = 13, fancybox = True,
           shadow = True, ncol = 2,
           loc = 'upper center', 
           bbox_to_anchor = (1.32, 1.17))
plt.xlabel('z-axis (m)', 
            fontsize = 13)
plt.ylabel('H$_{2}$ uptake (mol/kg)',
            fontsize = 13)
plt.grid(ls = '--')
plt.savefig('res_q1_profile_H2.png', dpi = 150, bbox_inches = 'tight')

# %%
# mole fraction
# %%
#### y1 Profile ####
plt.figure()
C_ov = y_res[:, 0*N:1*N]+y_res[:, 1*N:2*N]+y_res[:, 2*N:3*N]+y_res[:, 3*N:4*N]
cc = 0
for ii, tt in zip(ii_arr, t_sample):
    C_samp = y_res[ii,0*N:1*N]/C_ov[ii,:]
    plt.plot(z_dom, C_samp, 'k',
             linestyle = ls_list[cc%len(ls_list)],
             label = f't = {tt}'
             )
    cc += 1
plt.legend(fontsize = 13, fancybox = True,
           shadow = True, ncol = 2,
           loc = 'upper center', 
           bbox_to_anchor = (1.36, 0.92))
plt.xlabel('z-axis (m)', 
            fontsize = 13)
plt.ylabel('H$_{2}$ mole fraction (mol/mol)',
            fontsize = 13)
plt.grid(ls = '--')
plt.savefig('res_y1_profile_H2.png', dpi = 150, bbox_inches = 'tight')

#### y2 Profile ####
plt.figure()
C_ov = y_res[:, 0*N:1*N]+y_res[:, 1*N:2*N]+y_res[:, 2*N:3*N]+y_res[:, 3*N:4*N]
cc = 0
for ii, tt in zip(ii_arr, t_sample):
    C_samp = y_res[ii,1*N:2*N]/C_ov[ii,:]
    plt.plot(z_dom, C_samp, 'k',
             linestyle = ls_list[cc%len(ls_list)],
             label = f't = {tt}'
             )
    cc += 1
plt.legend(fontsize = 13, fancybox = True,
           shadow = True, ncol = 2,
           loc = 'upper center', 
           bbox_to_anchor = (1.36, 0.92))
plt.xlabel('z-axis (m)', 
            fontsize = 13)
plt.ylabel('CO mole fraction (mol/mol)',
            fontsize = 13)
plt.grid(ls = '--')
plt.savefig('res_y2_profile_CO.png', dpi = 150, bbox_inches = 'tight')

#### y3 Profile ####
plt.figure()
C_ov = y_res[:, 0*N:1*N]+y_res[:, 1*N:2*N]+y_res[:, 2*N:3*N]+y_res[:, 3*N:4*N]
cc = 0
for ii, tt in zip(ii_arr, t_sample):
    C_samp = y_res[ii,2*N:3*N]/C_ov[ii,:]
    plt.plot(z_dom, C_samp, 'k',
             linestyle = ls_list[cc%len(ls_list)],
             label = f't = {tt}'
             )
    cc += 1
plt.legend(fontsize = 13, fancybox = True,
           shadow = True, ncol = 2,
           loc = 'upper center', 
           bbox_to_anchor = (1.36, 0.92))
plt.xlabel('z-axis (m)', 
            fontsize = 13)
plt.ylabel('H$_{2}$O mole fraction (mol/mol)',
            fontsize = 13)
plt.grid(ls = '--')
plt.savefig('res_y3_profile_H2O.png', dpi = 150, bbox_inches = 'tight')


#### y4 Profile ####
plt.figure()
C_ov = y_res[:, 0*N:1*N]+y_res[:, 1*N:2*N]+y_res[:, 2*N:3*N]+y_res[:, 3*N:4*N]
cc = 0
for ii, tt in zip(ii_arr, t_sample):
    C_samp = y_res[ii,3*N:4*N]/C_ov[ii,:]
    plt.plot(z_dom, C_samp, 'k',
             linestyle = ls_list[cc%len(ls_list)],
             label = f't = {tt}'
             )
    cc += 1
plt.legend(fontsize = 13, fancybox = True,
           shadow = True, ncol = 2,
           loc = 'upper center', 
           bbox_to_anchor = (1.36, 0.92))
plt.xlabel('z-axis (m)', 
            fontsize = 13)
plt.ylabel('CO$_{2}$ mole fraction (mol/mol)',
            fontsize = 13)
plt.grid(ls = '--')
plt.savefig('res_y4_profile_CO2.png', dpi = 150, bbox_inches = 'tight')


'''