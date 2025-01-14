
# %%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
# %%
# Spatial setting for z-axis
# %%
# Number of spatial node
N = 101
L = 0.2
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

# Test of Ergun eqn function
dp_test = 1E-2  # 0.01 (m) = 1cm
mu_test = 1.8E-5 # (Pa s)
Mw1_test = 28/1E3   # kg/mol
Mw2_test = 44/1E3   # kg/mol
C1_test = np.logspace(np.log10(10),np.log10(30),N)
C2_test = np.zeros_like(C1_test)
C2_test[:80] = np.linspace(50,5,80)
C2_test[80:] = np.linspace(5,10, N-80)
C1_mid = (C1_test[1:] + C1_test[:-1])/2
C2_mid = (C2_test[1:] + C2_test[:-1])/2
rho_g_test = C1_mid*Mw1_test + C2_mid*Mw2_test
C_ov = C1_test + C2_test
P_ov_test = C_ov*8.3145*300
epsi_test = 0.3
u_test,_,_= Ergun(P_ov_test,z_dom, dp_test,mu_test, rho_g_test,epsi_test)
plt.figure(dpi=300)
plt.plot(z_dom[1:]-0.5*z_dom[1], u_test, 'o')
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
qm1 = 0.1 # H2
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
T_init = 300
R_gas = 8.3145
# Gas phase
C1_init = 0.5*P_init/R_gas/T_init*np.ones([N,])
C2_init = 0.5*P_init/R_gas/T_init*np.ones([N,])
C3_init = 0.0*P_init/R_gas/T_init*np.ones([N,])
C4_init = 0.0*P_init/R_gas/T_init*np.ones([N,])
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
Cv_out = 1E-2 
u_feed = 0.01

P_feed = 3.5    # (bar)

# Inlet conditions
T_feed = T_init
C1_feed = 0.2*P_feed*1E5/R_gas/T_feed # H2  (prod)
C2_feed = 0.3*P_feed*1E5/R_gas/T_feed # CO  (reac)
C3_feed = 0.3*P_feed*1E5/R_gas/T_feed # H2O (reac)
C4_feed = 0.2*P_feed*1E5/R_gas/T_feed # CO2 (prod)

# %%
# Model for odeint
# %%
# Additional parameters
D_AB = 1E-8
h = h_arr[0]
T_g = T_init
#u_g = 0.01      # (m/s) : advective velocity

rho_s = 1000    # (kg/m^3)
epsi = 0.35     # voide fraction of pellet
#k1, k2 = [0.2,0.2]
k1, k2, k3, k4 = [0.005,]*4
d_particle = 2E-3   # (m) : pellet particle size
mu_gas = 1.8E-5 # (Pa s) : viscosity of gas (air)  
Mw_gas = np.array([2, 28, 18, 44])*1E-3 # (kg/mol) : molar weight for each component

# Backflow at z = L 
C1_L = 0.9*P_end*1E5/R_gas/T_g  # H2 (prod)
C2_L = 0.05*P_end*1E5/R_gas/T_g # CO (reac)
C3_L = 0.05*P_end*1E5/R_gas/T_g # H2O (reac)
C4_L = 0*P_end*1E5/R_gas/T_g    # CO2 (prod)


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
    r1 = 0.01*np.ones([N,])
    r2 = 0.01*np.ones([N,])

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
    dC1dt = D_AB*ddC1 - duC1dz - rho_s*(1-epsi)/epsi*dq1dt+r1+r2 #H2
    dC2dt = D_AB*ddC2 - duC2dz - rho_s*(1-epsi)/epsi*dq2dt-r1-r2 #CO
    dC3dt = D_AB*ddC3 - duC3dz - rho_s*(1-epsi)/epsi*dq3dt-r1-r2 #H2O
    dC4dt = D_AB*ddC4 - duC4dz - rho_s*(1-epsi)/epsi*dq4dt+r1+r2 #CO2

    # Boundary conditions
    dC1dt[0] = - duC1dz[0] - rho_s*(1-epsi)/epsi*dq1dt[0] +r1[0]+r2[0]
    dC2dt[0] = - duC2dz[0] - rho_s*(1-epsi)/epsi*dq2dt[0] -r1[0]-r2[0]
    dC3dt[0] = - duC3dz[0] - rho_s*(1-epsi)/epsi*dq3dt[0] -r1[0]-r2[0]
    dC4dt[0] = - duC4dz[0] - rho_s*(1-epsi)/epsi*dq4dt[0] +r1[0]+r2[0]

    dC1dt[-1] = - duC1dz[-1] - rho_s*(1-epsi)/epsi*dq1dt[-1] +r1[-1]+r2[-1]
    dC2dt[-1] = - duC2dz[-1] - rho_s*(1-epsi)/epsi*dq2dt[-1] -r1[-1]-r2[-1]
    dC3dt[-1] = - duC3dz[-1] - rho_s*(1-epsi)/epsi*dq3dt[-1] -r1[-1]-r2[-1]
    dC4dt[-1] = - duC4dz[-1] - rho_s*(1-epsi)/epsi*dq4dt[-1] +r1[-1]+r2[-1]
    
    dydt = np.concatenate([dC1dt,dC2dt, dC3dt, dC4dt, dq1dt, dq2dt, dq3dt,dq4dt])
    return dydt

# %%
# Run 
# %%
t_ran = np.arange(0,20+0.0025,0.0025)
y_res = odeint(model_col, y0, t_ran )
print(y_res.shape)
# %%
# Graph drawing
# %%
t_sample = t_ran[::400]
ii_arr = np.arange(len(t_ran))[::400]

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

# C4 Profile
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
plt.ylabel('Concentration (mol/m$^3$)',
            fontsize = 13)
plt.grid(ls = '--')
plt.savefig('res_C4_profile.png', dpi = 150, bbox_inches = 'tight')

# q4 Profile
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
plt.ylabel('Uptake (mol/kg)',
            fontsize = 13)
plt.grid(ls = '--')
plt.savefig('res_q4_profile.png', dpi = 150, bbox_inches = 'tight')


# %%

