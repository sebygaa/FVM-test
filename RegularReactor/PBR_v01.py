# %%
# Importing packages

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# %%
# Key parameters

# %%
NN = 1000         # Number of points in the reactor
rho_cat = 1000  # (kg/m^3)
epsi = 0.4      # void fraction
u = 0.05        # (m/s) Advective velocity

# %%
# Function to calculate the rate of reaction
# %%
def rate_cal(k1,k2, P1,P2,P3,P4):
    r1_ret = k1*P2*P3
    r2_ret = k2*P1*P4
    return r1_ret, r2_ret

# %%
# Function for the ODEs

# %%
R_gas = 8.314 # J/mol.K
######## TEMPERATURE DEPENDENT EQUILIBRIUM CONSTANT ########
T = 773 # K # Temperature (400 oC)
K_eq = 10**(-2.4198 + 0.0003855*T + 2180.6/T) # Equilibrium constant
#############################################################

######## REACTION RATE CONSTANTS ########
k1_wgs = 0.005
k2_wgs = k1_wgs/K_eq
#########################################
def PBR(y, z):
    C1 = y[0]
    C2 = y[1]
    C3 = y[2]
    C4 = y[3]
    # Concentration to Pressure (bar)
    P1 = C1*R_gas*T/1E5
    P2 = C2*R_gas*T/1E5
    P3 = C3*R_gas*T/1E5
    P4 = C4*R_gas*T/1E5

    # Reaction rate calculation
    r1,r2 = rate_cal(k1_wgs, k2_wgs, P1,P2,P3,P4)
    # ODEs
    dC1dz = 1/u*(1-epsi)/epsi*rho_cat*(r1-r2)
    dC2dz = 1/u*(1-epsi)/epsi*rho_cat*(-r1+r2)
    dC3dz = 1/u*(1-epsi)/epsi*rho_cat*(-r1+r2)
    dC4dz = 1/u*(1-epsi)/epsi*rho_cat*(r1-r2)
    return np.array([dC1dz, dC2dz, dC3dz, dC4dz])

# %%
# Initial conditions (Feed condition)
# %%
y_feed = np.array([0, 0.5, 0.5, 0]) # H2, CO, H2O, CO2
#y1_feed,y2_feed,y3_feed,y4_feed = 0, 0.5, 0.5, 0   # H2, CO, H2O, CO2 
P_feed = 1                          # (bar)
C_feed = y_feed*P_feed*1E5/R_gas/T  # (mol/m^3)
print(C_feed)

# %%
# Solve the ODEs
# %%
z = np.linspace(0,2,NN)
C_res = odeint(PBR, C_feed, z)

# %%
# Plot the results
# %%
plt.figure()
plt.plot(z, C_res[:,0], label='H$_{2}$', ls = '-')
plt.plot(z, C_res[:,1], label='CO', ls = '--')
plt.plot(z, C_res[:,2], label='H$_{2}$O', ls = '--')
plt.plot(z, C_res[:,3], label='CO$_{2}$', ls = '-')

plt.xlabel('Position (m)', fontsize = 12)
plt.ylabel('Concentration (mol/m^3)', fontsize = 12)
plt.legend(fontsize=13)

# %%
# Mole fraction graph
# %%
plt.figure()
C_ov = np.sum(C_res, axis=1)
plt.plot(z, C_res[:,0]/C_ov, label='H$_{2}$', ls = '-')
plt.plot(z, C_res[:,1]/C_ov, label='CO', ls = '--')
plt.plot(z, C_res[:,2]/C_ov, label='H$_{2}$O', ls = '--')
plt.plot(z, C_res[:,3]/C_ov, label='CO$_{2}$', ls = '-')

plt.xlabel('Position (m)', fontsize = 12)
plt.ylabel('Mole fraction (mol/mol)', fontsize = 12)
plt.legend(fontsize=13)
print('mole fraction of H2 at the exit:', C_res[-1,0]/C_ov[-1])
print()
print('Conversion of CO:', (C_feed[1] - C_res[-1,1])/C_feed[1]*100)
#print(C_res[-1,0]/C_ov[-1])
# %%
# Overall Conversion calculation

# %%
# Based on the feed CO concentration
X_CO = (C_feed[1] - C_res[-1,1])/C_feed[1]
print('The overall conversion of CO is:', X_CO*100, '%')

# %%
