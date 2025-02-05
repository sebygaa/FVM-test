# %%
# Importing packages

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# %%
# Key parameters
# %%
NN = 201       # Number of points in the reactor
L_bed = 0.2     # Catalyst Bed length
rho_cat = 1000  # (kg/m^3)
epsi = 0.4      # void fraction
u_feed = 0.15        # (m/s) Advective velocity

# %%
# Function to calculate the rate of reaction
# %%
# Comp 1 ~ 4 =
# CO2, H2, CO, CH3OH, H2O
# Reactions:
# r1: CO + 2H2 <-> CH3OH 
# r2: CO2 + H2 <-> CO + H2O
# r3: CO + 3H2 <-> CH3OH + H2O
# r_ov = -2*r1+2*r1_rev -2*r3+2*r3_rev

def rate_cal(k1, k2, k3, 
             k1_rev, k2_rev, k3_rev, 
             P1, P2, P3, P4, P5):
    r1 = k1 * P3 * P2**2
    r1_rev = k1_rev * P4
    r2 = k2 * P1 * P2
    r2_rev = k2_rev * P3 * P5
    r3 = k3 * P1 * P2**3
    r3_rev = k3_rev * P4 * P5
    return r1, r2, r3, r1_rev, r2_rev, r3_rev

# %%
# Function for the ODEs

# %%
R_gas = 8.314 # J/mol.K
######## TEMPERATURE DEPENDENT EQUILIBRIUM CONSTANT ########
T = 773 # K # Temperature (400 oC)
#K_eq = 10**(-2.4198 + 0.0003855*T + 2180.6/T) # Equilibrium constant
#############################################################

######## REACTION RATE CONSTANTS ########
# Comp 1 ~ 4 =
# CO2, H2, CO, CH3OH, H2O
# Reactions:
# r1: CO + 2H2 <-> CH3OH 
# r2: CO2 + H2 <-> CO + H2O
# r3: CO + 3H2 <-> CH3OH + H2O
# r_ov = -2*r1+2*r1_rev -2*r3+2*r3_rev

k1 = 0.006
k2 = 0.002
k3 = 0.004
k1_rev = 0.003
k2_rev = 0.004
k3_rev = 0.001
k_rxn_list = [k1,k2,k3,
              k1_rev,k2_rev,k3_rev]
#########################################

def PBR(y, z, k_list):
    C1 = y[0]   # CO2
    C2 = y[1]   # H2
    C3 = y[2]   # CO
    C4 = y[3]   # CH3OH
    C5 = y[4]   # H2O
    u = y[5]    # velocity
    C_ov = C1+C2+C3+C4+C5
    y1,y2,y3,y4,y5 = np.array(y[:5])/C_ov
    k1, k2, k3, k1_rev, k2_rev, k3_rev = k_list
    # Concentration to Pressure (bar)
    P1 = C1*R_gas*T/1E5
    P2 = C2*R_gas*T/1E5
    P3 = C3*R_gas*T/1E5
    P4 = C4*R_gas*T/1E5
    P5 = C5*R_gas*T/1E5

    # Reaction rate calculation
    #r1,r2 = rate_cal(k1_wgs, k2_wgs, P1,P2,P3,P4)
    r1,r2,r3, r1_rev, r2_rev, r3_rev = rate_cal(k1,k2,k3,
                                                k1_rev,k2_rev,k3_rev,
                                                P1,P2,P3,P4,P5)
    # Reaction terms
    Sig_CO2= -(r2-r2_rev)
    Sig_H2 =-2*(r1-r1_rev)-(r2-r2_rev)-3*(r3-r3_rev)
    Sig_CO = -(r1-r1_rev) + (r2-r2_rev) - (r3-r3_rev)
    Sig_CH3OH = (r1-r1_rev)+(r3-r3_rev)
    Sig_H2O= (r2-r2_rev)+(r3-r3_rev)
    r_ov = Sig_CO2 + Sig_H2 + Sig_CO + Sig_CH3OH + Sig_H2O

    # ODEs
    term_r_ov = 1/u*(1-epsi)/epsi*rho_cat*r_ov
    dC1dz = -y1*term_r_ov+ 1/u*(1-epsi)/epsi*rho_cat*Sig_CO2   # CO2
    dC2dz = -y2*term_r_ov+ 1/u*(1-epsi)/epsi*rho_cat*Sig_H2    # H2
    dC3dz = -y3*term_r_ov+ 1/u*(1-epsi)/epsi*rho_cat*Sig_CO    # CO
    dC4dz = -y4*term_r_ov+ 1/u*(1-epsi)/epsi*rho_cat*Sig_CH3OH # CH3OH
    dC5dz = -y5*term_r_ov+ 1/u*(1-epsi)/epsi*rho_cat*Sig_H2O   # H2O
    dudz = 1/C_ov*(1-epsi)/epsi*rho_cat*r_ov 
    
    return np.array([dC1dz, dC2dz, dC3dz, dC4dz, dC5dz, dudz])

# %%
# Initial conditions (Feed condition)
# %%
y_feed = np.array([0.2, 0.6, 0.2, 0, 0]) # CO2, H2, CO, CH3OH, H2O
#y1_feed,y2_feed,y3_feed,y4_feed = 0, 0.5, 0.5, 0   # H2, CO, H2O, CO2 
P_feed = 8                          # (bar)
C_feed = y_feed*P_feed*1E5/R_gas/T  # (mol/m^3)
C_u_feed = np.concatenate([C_feed, [u_feed]])
print(C_u_feed)

# %%
# Solve the ODEs
# %%
z = np.linspace(0,L_bed,NN)
C_res = odeint(PBR, C_u_feed, z, args = (k_rxn_list,))

# %%
# Plot the results
# %%
plt.figure()
plt.plot(z, C_res[:,0], label='CO$_{2}$', ls = ':')
plt.plot(z, C_res[:,1], label='H$_{2}$', ls = '-.')
plt.plot(z, C_res[:,2], label='CO', ls = '--')
plt.plot(z, C_res[:,3], label='CH$_{3}$OH', ls = '-')
plt.plot(z, C_res[:,4], label='H$_{2}$O', ls = '-')

plt.xlabel('Position (m)', fontsize = 12)
plt.ylabel('Concentration (mol/m$^{3}$)', fontsize = 12)
plt.legend(fontsize=13)
plt.show()

# %%
# Mole fraction graph
# %%
plt.figure()
C_ov = np.sum(C_res[:,:5], axis=1)
plt.plot(z, C_res[:,0]/C_ov, label='CO$_{2}$', ls = ':')
plt.plot(z, C_res[:,1]/C_ov, label='H$_{2}$', ls = '-.')
plt.plot(z, C_res[:,2]/C_ov, label='CO', ls = '--')
plt.plot(z, C_res[:,3]/C_ov, label='CH$_{3}$OH', ls = '-')
plt.plot(z, C_res[:,4]/C_ov, label='H$_{2}$O', ls = '-')

plt.xlabel('Position (m)', fontsize = 12)
plt.ylabel('Mole fraction (mol/mol)', fontsize = 12)
plt.legend(fontsize=13)
print('mole fraction of H2 at the exit:', C_res[-1,0]/C_ov[-1])
print()
print('Conversion of CO:', (C_feed[1] - C_res[-1,1])/C_feed[1]*100)
#print(C_res[-1,0]/C_ov[-1])

# %%
plt.figure()
plt.plot(z, C_res[:,-1], 'k-',
         label = 'velocity (m/s)')
plt.xlabel('Position (m)', fontsize = 12)
plt.ylabel('Advective velocity (m/s)', fontsize = 12)
plt.legend(fontsize=13)

# %%
# Overall Conversion calculation

# %%
# Based on the feed CO concentration
X_CO = (C_feed[2] - C_res[-1,2])/C_feed[2]
print('The overall conversion of CO is:', X_CO*100, '%')

# %%
# Dummy data for testing parameter estimation

# %%
T_feed = T
u_feed = u_feed
NN = NN
L_bed =L_bed
k1 = 0.006
k2 = 0.002
k3 = 0.004
k1_rev = 0.003
k2_rev = 0.004
k3_rev = 0.001
k_rxn_list = [k1,k2,k3,
              k1_rev,k2_rev,k3_rev]
def P_2_X(P_list,k_list):
    P1,P2,P3,P4,P5 = P_list
    # CO2, H2, CO, CH3OH, H2O
    #C1,C2,C3,C4,C5 = np.array(P_list)/R_gas/T_feed*1E5
    C_feed = np.array(P_list)/R_gas/T_feed*1E5
    C_u_feed = np.concatenate([C_feed, [u_feed,]])
    z_dom = np.linspace(0,L_bed, NN)
    C_res = odeint(PBR, C_u_feed, z_dom, args=(k_list,))
    u_end = C_res[-1,-1]
    X_CO2 = (u_feed*C_feed[0] - u_end*C_res[-1,0] )/u_feed/C_feed[0]
    X_H2 = (u_feed*C_feed[1] - u_end*C_res[-1,1] )/u_feed/C_feed[1]
    X_CO = (u_feed*C_feed[2] - u_end*C_res[-1,2] )/u_feed/C_feed[2]
    
    return X_CO2,X_H2, X_CO
    
P_f_list_test = [2,4,2,0,0]
X_test = P_2_X(P_f_list_test, k_rxn_list) 
print(X_test)

# %%
P_ov = 10
P_CO2_arr = np.arange(0.5,5+0.5, 0.5)
P_H2_arr = np.arange(0.5,5+0.5, 0.5)

P_CO2_input = []
P_H2_input = []
P_CO_input = []
X_H2_output = []
X_CO_output = []

for pco2 in P_CO2_arr:
    for ph2 in P_H2_arr:
        pco = P_ov-pco2-ph2
        if pco <= 0:
            break
        else:
            P_feed_list = [pco2,ph2,pco, 0, 0]
            xco2,xh2,xco = P_2_X(P_feed_list, k_rxn_list)
            P_CO2_input.append(pco2)
            P_H2_input.append(ph2)
            P_CO_input.append(pco)
            X_H2_output.append(xh2)
            X_CO_output.append(xco)
di = {'P_CO2': P_CO2_input,
      'P_H2': P_H2_input,
      'P_CO': P_CO_input,
      'X_H2': X_H2_output, 
      'X_CO': X_CO_output,}
import pandas as pd
df = pd.DataFrame(di)
print(df)
# %%
# If k1 is missing...
# %%
k2 = 0.002
k3 = 0.004
k1_rev = 0.003
k2_rev = 0.004
k3_rev = 0.001
def obj(k1_guess,):
    k_list_tmp = [k1_guess[0], k2,k3,k1_rev,k2_rev,k3_rev]
    X_H2_list = []
    X_CO_list = []
    for pco2, ph2, pco in zip(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2]):
        P_list_tmp = [pco2,ph2,pco,0,0]
        
        _, xh2,xco = P_2_X(P_list_tmp, k_list_tmp)
        X_H2_list.append(xh2)
        X_CO_list.append(xco)
    X_H2_arr = np.array(X_H2_list)
    X_CO_arr = np.array(X_CO_list)
    diff_sq = (X_H2_arr - X_H2_output)**2 + (X_CO_arr - X_CO_output)**2
    diff_sq_sum = np.sum(diff_sq)
    return diff_sq_sum

# %%
k_guess_list = np.linspace(0.001, 0.015, 30)

k1_sol = [0.006,]
diff_sq_sol = obj(k1_sol)

diff_sq_list = []
for kk in k_guess_list:
    diff_sq_tmp = obj([kk,])
    diff_sq_list.append(diff_sq_tmp)

# %%
plt.plot(k_guess_list, diff_sq_list,
         'k-', linewidth = 1.8 )
plt.plot([k1_sol],[diff_sq_sol], 'o',
         ms = 9, mfc = 'r', mec = 'k', mew = 1.5)
plt.xlabel('k guess')
plt.ylabel('Mean squared error (MSE)')

# %%

# %%
from scipy.optimize import minimize

# %%
# Parameter estimation with optim. solver
# %%
k1_guess0 = [0.009,]
#opt_res = minimize(obj, k1_guess0, method = 'Nelder-mead')
#opt_res = minimize(obj, k1_guess0, method = 'BFGS')
#opt_res = minimize(obj, k1_guess0,)
opt_res = minimize(obj, k1_guess0, method = 'Nelder-mead')

# %%
# Solution
# %%
print(opt_res)
print()
print('[SOLUTION of k fitting]')
print('k1 = ', opt_res.x[0])
# %%
MSE_opt = obj(opt_res.x)
plt.plot(k_guess_list, diff_sq_list,
         'k-', linewidth = 1.8 )
plt.title('k1 from Parameter estimation', fontsize = 13.5)
plt.plot([opt_res.x[0]],[MSE_opt], 'o',
         ms = 9, mfc = 'r', mec = 'k', mew = 1.5)
plt.xlabel('k guess')
plt.ylabel('Mean squared error (MSE)')
