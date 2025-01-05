
# %%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
# %%
# Spatial setting for z-axis
# %%
# Number of spatial node
N = 251
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
def Ergun_bi(C_list, T_gas, z,
              d_p, mu, Mw_list, void_frac):
    C1, C2 = C_list
    Mw1,Mw2 = Mw_list
    C1_mid = (C1[1:] + C1[:-1])/2
    C2_mid = (C2[1:] + C2[:-1])/2
    rho_g_tmp = C1_mid*Mw1 + C2_mid*Mw2
    C_ov_tmp = C1 + C2
    P_ov_tmp = C_ov_tmp*R_gas*T_gas
    u_ret, arg_u_pos, arg_u_neg = Ergun(P_ov_tmp, 
                                        z, d_p, mu, rho_g_tmp, void_frac)
    return u_ret, arg_u_pos, arg_u_neg


# %%
# Isotherm Model 
# %%
qm1 = 0.5
qm2 = 3
b1 = 0.01
b2 = 0.2
def iso(P1,P2):
    deno = 1+b1*P1 + b2*P2
    nume1 = qm1*b1*P1
    nume2 = qm2*b2*P2
    q1 = nume1/deno
    q2 = nume2/deno
    return q1, q2

# %%
# Initial Values
# %%
# Initial P, T, C1, C2, q1, q2, y0
P_init = 1.5E5*np.ones([N,])    # (Pa)
T_init = 300
R_gas = 8.3145
C1_init = 1*P_init/R_gas/T_init*np.ones([N,])
C2_init = 0*P_init/R_gas/T_init*np.ones([N,])
q1_init, q2_init = iso(1*P_init*1E-5, 0*P_init*1E-5)
y0 = np.concatenate([C1_init, C2_init, q1_init, q2_init])

# %%
# Boundary Conditions
# %%
# Pressure conditions
P_end = 1       # (bar)
Cv_out = 1E-2 

P_feed = 1.5    # (bar)
# Inlet conditions
T_feed = T_init
C1_feed = 0.5*P_feed*1E5/R_gas/T_feed
C2_feed = 0.5*P_feed*1E5/R_gas/T_feed

# %%
# Model for odeint
# %%
# Additional parameters
D_AB = 1E-6
h = h_arr[0]
T_g = T_init
#u_g = 0.01      # (m/s) : advective velocity
u_feed = 0.00
rho_s = 1000    # (kg/m^3)
epsi = 0.35     # voide fraction of pellet
k1, k2 = [0.2,0.2]
d_particle = 2E-3   # (m) : pellet particle size
mu_gas = 1.8E-5 # (Pa s) : viscosity of gas (air)  
Mw_gas = np.array([44,28])*1E-3 # (kg/mol) : molar weight for each component


# PDE -> ODE model
def model_col(y,T ):
    C1 = y[:N]
    C2 = y[1*N : 2*N]
    q1 = y[2*N : 3*N]
    q2 = y[3*N : 4*N]

    # Pressure
    P1 = C1*R_gas*T_g/1E5   # in bar
    P2 = C2*R_gas*T_g/1E5   # in bar
    
    # velocity from Ergun equation
    u_g, arg_u_posi, arg_u_nega = Ergun_bi([C1,C2], T_g, z_dom,
                                           d_particle, mu_gas,
                                           Mw_gas, epsi)
    u_g = np.concatenate([[0],u_g])
    du_g = (u_g[1:] - u_g[:-1])/h
    du_g = np.concatenate([du_g,[0]])

    # Valve equation
    P_ov_end = P1[-1]+P2[-1]
    u_out = Cv_out*(P_ov_end - P_end) # (bar) -> m/s
    # Isotherm
    q1sta, q2sta = iso(P1,P2,)

    # Discretization
    dC1 = d@C1
    dC2 = d@C2
    ddC1 = dd@C1
    ddC2 = dd@C2

    # LDF
    dq1dt = k1*(q1sta - q1)
    dq2dt = k2*(q2sta - q2)

    # Mass balanace
    dC1dt = D_AB*ddC1 - u_g*dC1 - du_g*C1 - rho_s*(1-epsi)/epsi*dq1dt
    dC2dt = D_AB*ddC2 - u_g*dC2 - du_g*C2 - rho_s*(1-epsi)/epsi*dq2dt

    # Boundary conditions
    #dC1dt[0] = D_AB*(C1[1]-2*C1[0]+C1_feed)/h**2 - (u_g[1]*C1[0] - u_feed*C1_feed)/h - rho_s*(1-epsi)/epsi*dq1dt[0]
    #dC2dt[0] = D_AB*(C2[1]-2*C2[0]+C2_feed)/h**2 - (u_g[1]*C2[0] - u_feed*C2_feed)/h - rho_s*(1-epsi)/epsi*dq2dt[0]
    #dC1dt[-1] = D_AB*(C1[-2]-2*C1[-1]+C1[-2])/h**2 - (u_out*C1[-1] - u_g[-2]*C1[-2])/h - rho_s*(1-epsi)/epsi*dq1dt[-1]
    #dC2dt[-1] = D_AB*(C2[-2]-2*C2[-1]+C2[-2])/h**2 - (u_out*C2[-1] - u_g[-2]*C2[-2])/h - rho_s*(1-epsi)/epsi*dq2dt[-1]
    
    dC1dt[0] = - (u_g[1]*C1[0] - u_feed*C1_feed)/h - rho_s*(1-epsi)/epsi*dq1dt[0]
    dC2dt[0] = - (u_g[1]*C2[0] - u_feed*C2_feed)/h - rho_s*(1-epsi)/epsi*dq2dt[0]

    dC1dt[-1] = - (u_out*C1[-1] - u_g[-1]*C1[-2])/h - rho_s*(1-epsi)/epsi*dq1dt[-1]
    dC2dt[-1] = - (u_out*C2[-1] - u_g[-1]*C2[-2])/h - rho_s*(1-epsi)/epsi*dq2dt[-1]
    
    dydt = np.concatenate([dC1dt,dC2dt, dq1dt, dq2dt])
    return dydt

# %%
# Run 
# %%
t_ran = np.arange(0,10+0.05,0.05)
y_res = odeint(model_col, y0, t_ran )
print(y_res.shape)
# %%
# Graph drawing
# %%
t_sample = t_ran[::20]
ii_arr = np.arange(len(t_ran))[::20]

ls_list = ['-','--','-.',':']
cc = 0
for ii, tt in zip(ii_arr, t_sample):
    C_samp = y_res[ii, 0*N:1*N]
    plt.plot(z_dom, C_samp, 'k',
             linestyle = ls_list[cc%len(ls_list)],
             label = f't = {tt}'
             )
    cc += 1
plt.legend(fontsize = 13, fancybox = True,
           shadow = True, ncol = 2,
           loc = 'upper center', 
           bbox_to_anchor = (0.92, 0.97))
# %%
