# %%
# Importing the required libraries
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
    
# %%
# Ergun equation
# %% 
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
# Class definition
mu_gas = 1.8E-5
Mw_gas = np.array([2,28, 18, 44])*1E-3
R_gas = 8.314 # J/mol/K
D_AB = 1E-8 # m2/s

class AdsCatColumn:
    # 1) Define the AdsCatColumn class
    def __init__(self, L, N, ):
        try:
            if not self.is_init:
                print("The object is already initialized")
                self.is_init_re = True
        except:
            self.is_init = True
            self.is_init_re = False

        self.L = L
        self.N = N
        self.z = np.linspace(0, L, N)
        self.Mat_rev = np.zeros([N,N])
        for ii in range(N):
            self.Mat_rev[ii, -ii-1] = 1

        h_arr = (self.z[1] - self.z[0]) * np.ones(N)
        dd_arr = 1/h_arr**2
        dd_upp = np.diag(dd_arr[:-1],1)
        dd_mid = np.diag(dd_arr,0)
        dd_low = np.diag(dd_arr[1:],-1)
        dd = dd_upp -2*dd_mid + dd_low
        dd[0,:] = 0
        dd[-1,:] = 0
        self.dd = dd
        # Check list for run
        self.is_ads_info = False
        self.is_cat_info = False
        self.is_pac_info = False
        self.is_feed = False
        self.is_outlet = False
        self.is_init = False
        self.forward_flow = True

    # 2) Define the packing geometry
    def pac_info(self, epsi, x_cat, d_particle, ):
        self.epsi = epsi
        self.x_cat = x_cat
        self.d_particle = d_particle
        self.is_pac_info = True
    # 3) Define the adsorbent information
    def ads_info(self, iso, k_list, rho_ads=1000, ):
        self.iso =iso
        self.k_list = k_list
        self.rho_ads = rho_ads
        self.is_ads_info = True

    # 4) Define the catalyst information
    def cat_info(self, k_f_ref, T_ref, E_a_f,rho_cat=1000, orders= [1,1,1,1] ):
        self.k_f_ref = k_f_ref
        self.T_ref = T_ref
        self.E_a_f = E_a_f
        self.rho_cat = rho_cat
        self.orders = orders
        self.is_cat_info = True
    
    # 5-1) Define the feed conditions
    def feed_condi_y(self, y_feed_list, P_feed, T_feed, u_feed=0, Cv_in = 0.01, const_u = True, ):
        self.y_feed = y_feed_list
        self.T_feed = T_feed
        self.P_feed = P_feed
        self.u_feed = u_feed
        C_feed_list = []
        for yy in y_feed_list:
            C_tmp = yy*P_feed*1E5/R_gas/T_feed
            C_feed_list.append(C_tmp)
        self.C_feed = C_feed_list

        self.const_u = const_u
        self.Cv_in = Cv_in
        self.is_feed = True

    # 5-2) Define the feed conditions
    def feed_condi_C(self, C_feed_list, T_feed, u_feed=0, Cv_in = 0.01, const_u = True, ):
        self.C_feed = C_feed_list
        self.T_feed = T_feed
        P_feed = np.sum(C_feed_list)*R_gas*T_feed/1E5 # in (bar)
        self.P_feed = P_feed # in (bar)
        self.u_feed = u_feed
        #C_feed_list = []
        #for yy in y_feed_list:
        #    C_tmp = yy*P_feed*1E5/R_gas/T_feed
        #    C_feed_list.append(C_tmp)
        #self.C_feed = C_feed_list

        self.const_u = const_u
        self.Cv_in = Cv_in
        self.is_feed = True
    
    # 6) Define the outlet conditions
    def outlet_condi_y(self, y_end_list, P_end, T_end, Cv_end, ):
        self.y_end = y_end_list
        self.T_end = T_end
        self.P_end = P_end
        C_end_list = []
        for yy in y_end_list:
            C_tmp = yy*P_end*1E5/R_gas/T_end
            C_end_list.append(C_tmp)
        self.C_end = C_end_list
        
        self.Cv_end = Cv_end
        self.is_outlet = True

    # 7-1) Define initial conditions
    def init_condi_C(self, C_init, q_init, T_init, equil = False, ):
        self.C_init = C_init
        self.q_init = q_init
        self.T_init = T_init
        try:
            if equil:
                P_list = []
                for CC in C_init:
                    P_list.append(CC*R_gas*T_init/1E5)
                self.q_init = self.iso(*P_list, T_init)
        except:
            print('Isotherm model is not defined')
        self.is_init = True

    # 7-2) Define initial conditions
    def init_condi_y(self, P_init, y_init, q_init, T_init, equil = False, ):
        #self.C_init = C_init
        C1_init = y_init[0]*P_init*1E5/8.3145/T_init
        C2_init = y_init[1]*P_init*1E5/8.3145/T_init
        C3_init = y_init[2]*P_init*1E5/8.3145/T_init
        C4_init = y_init[3]*P_init*1E5/8.3145/T_init
        self.C_init = [C1_init, C2_init, C3_init, C4_init,]
        self.q_init = q_init
        self.T_init = T_init
        try:
            if equil:
                P_list = []
                for CC in self.C_init:
                    P_list.append(CC*R_gas*T_init/1E5)
                self.q_init = self.iso(*P_list, T_init)
            elif q_init == None:
                P_list = []
                for CC in self.C_init:
                    P_list.append(CC*R_gas*T_init/1E5)
                self.q_init = self.iso(*P_list, T_init)
        except:
            print('Isotherm model is not defined')
        self.is_init = True

    # 8) Flow directions
    def Flow_direction(self, forward = True):
        if forward:
            self.forward_flow = True
            return
        else:
            self.forward_flow = False
            return

    # 9) run the simulations
    def run_mamo(self, t_span, ):
        if not self.is_init:
            print("The object is not initialized")
            return
        if not self.is_ads_info:
            print("The adsorbent info is not defined")
            return
        if not self.is_cat_info:
            print("The catalyst info is not defined")
            return
        if not self.is_pac_info:
            print("The packing info is not defined")
            return
        if not self.is_feed:
            print("The feed info is not defined")
            return
        if not self.is_outlet:
            print("The outlet info is not defined")
            return
        if self.forward_flow:
            #print("Check the flow direction: Forward")
            y0 = np.concatenate([self.C_init[0], self.C_init[1],
                                 self.C_init[2], self.C_init[3],
                                 self.q_init[0], self.q_init[1], 
                                 self.q_init[2], self.q_init[3],]) 
        else:
            y0_list = []
            for ii in range(4):
                C_tmp = self.Mat_rev @ self.C_init[ii]
                y0_list.append(C_tmp)
            for ii in range(4):
                q_tmp = self.Mat_rev @ self.q_init[ii]
                y0_list.append(q_tmp)
            y0 = np.concatenate(y0_list)
        h = self.z[1]-self.z[0]
        D_AB = 1E-8
        
        k_r1 = self.k_f_ref*np.exp(-self.E_a_f/8.3145*(1/self.T_feed - 1/self.T_ref)) # Forward Reaction
        K_eq = np.exp(5693.5 / self.T_feed + 1.077 * np.log(self.T_feed) + 5.44e-4 * self.T_feed - 1.125e-7 * self.T_feed**2 - 49170 / self.T_feed**2 - 13.148) # Equilibrium constant
        
        k_r2 = k_r1/K_eq # Backward WGS reaction
        def model_col(y, t):
            C1 = y[:self.N]
            C2 = y[self.N:2*self.N]
            C3 = y[2*self.N:3*self.N]
            C4 = y[3*self.N:4*self.N]

            q1 = y[4*self.N:5*self.N]
            q2 = y[5*self.N:6*self.N]
            q3 = y[6*self.N:7*self.N]
            q4 = y[7*self.N:8*self.N]

            # Pressure
            P1 = C1*R_gas*self.T_feed/1E5   # bar
            P2 = C2*R_gas*self.T_feed/1E5   # bar
            P3 = C3*R_gas*self.T_feed/1E5   # bar
            P4 = C4*R_gas*self.T_feed/1E5   # bar

            # Velocity form Ergun equation
            #u_g, arg_u_posi, arg_u_nega = Ergun_qu([C1,C2,C3,C4], self.T_feed, )
            u_g, arg_u_posi, arg_u_nega = Ergun_qu([C1,C2,C3,C4], self.T_feed, self.z,
                                                self.d_particle, mu_gas,
                                                Mw_gas, self.epsi)

            # Valve equation
            P_ov_in = P1[0]+P2[0]+P3[0]+P4[0]
            P_ov_end = P1[-1]+P2[-1] + P3[-1]+ P4[-1]
            u_feed = self.const_u*self.u_feed + (1-self.const_u)*self.Cv_in*(self.P_feed - P_ov_in)
            u_out = self.Cv_end*(P_ov_end - self.P_end) # (bar) -> m/s

            # uC1
            uC1_tmp = u_g*C1[:-1]
            uC1_back = u_g*C1[1:]
            uC1_tmp[arg_u_nega] = uC1_back[arg_u_nega]
            uC1_z0 = self.C_feed[0]*np.max([u_feed, 0]) + C1[0]*np.min([u_feed, 0])
            uC1_zL = C1[-1]*np.max([u_out,0]) + self.C_end[0]*np.min([u_out, 0])
            uC1 = np.concatenate([ [uC1_z0,], uC1_tmp, [uC1_zL,], ] )
            duC1dz = (uC1[1:]-uC1[:-1])/h
            # uC2
            uC2_tmp = u_g*C2[:-1]
            uC2_back = u_g*C2[1:]
            uC2_tmp[arg_u_nega] = uC2_back[arg_u_nega]
            uC2_z0 = self.C_feed[1]*np.max([u_feed,0]) + C2[0]*np.min([u_feed, 0])
            uC2_zL = C2[-1]*np.max([u_out,0]) + self.C_end[1]*np.min([u_out, 0])
            uC2 = np.concatenate([[uC2_z0], uC2_tmp, [uC2_zL],])
            duC2dz = (uC2[1:]-uC2[:-1])/h
            # uC3
            uC3_tmp = u_g*C3[:-1]
            uC3_back = u_g*C3[1:]
            uC3_tmp[arg_u_nega] = uC3_back[arg_u_nega]
            uC3_z0 = self.C_feed[2]*np.max([u_feed,0]) + C3[0]*np.min([u_feed, 0])
            uC3_zL = C3[-1]*np.max([u_out,0]) + self.C_end[2]*np.min([u_out, 0])
            uC3 = np.concatenate([[uC3_z0], uC3_tmp, [uC3_zL],])
            duC3dz = (uC3[1:]-uC3[:-1])/h
            # uC4
            uC4_tmp = u_g*C4[:-1]
            uC4_back = u_g*C4[1:]
            uC4_tmp[arg_u_nega] = uC4_back[arg_u_nega]
            uC4_z0 = self.C_feed[3]*np.max([u_feed,0]) + C4[0]*np.min([u_feed, 0])
            uC4_zL = C4[-1]*np.max([u_out,0]) + self.C_end[3]*np.min([u_out, 0])
            uC4 = np.concatenate([[uC4_z0], uC4_tmp, [uC4_zL],])
            duC4dz = (uC4[1:]-uC4[:-1])/h

            # Isotherm
            q1sta, q2sta, q3sta, q4sta = self.iso(P1,P2,P3,P4, self.T_feed)

            # Reactions 
            # (You need to define reaction rate expressions here)
            r1 = k_r1*P2**self.orders[1]*P3**self.orders[2]
            r2 = k_r2*P1**self.orders[0]*P4**self.orders[3]

            # Discretization
            ddC1 = self.dd@C1
            ddC2 = self.dd@C2
            ddC3 = self.dd@C3
            ddC4 = self.dd@C4

            # LDF
            dq1dt = self.k_list[0] * (q1sta - q1)
            dq2dt = self.k_list[1] * (q2sta - q2)
            dq3dt = self.k_list[2] * (q3sta - q3)
            dq4dt = self.k_list[3] * (q4sta - q4)

            
            # Mass balance
            m_ad = self.rho_ads*(1-self.epsi)/self.epsi*(1-self.x_cat)  #흡착
            m_cat = self.rho_cat*(1-self.epsi)/self.epsi*self.x_cat     #반응
            dC1dt = D_AB*ddC1 - duC1dz - m_ad*dq1dt + m_cat*(r1-r2) #H2
            dC2dt = D_AB*ddC2 - duC2dz - m_ad*dq2dt + m_cat*(-r1+r2) #CO
            dC3dt = D_AB*ddC3 - duC3dz - m_ad*dq3dt + m_cat*(-r1+r2) #H2O
            dC4dt = D_AB*ddC4 - duC4dz - m_ad*dq4dt + m_cat*(r1-r2) #CO2

            # Boundary conditions
            dC1dt[0] = - duC1dz[0] - m_ad[0]*dq1dt[0] + m_cat[0]*(+r1[0]-r2[0])
            dC2dt[0] = - duC2dz[0] - m_ad[0]*dq2dt[0] + m_cat[0]*(-r1[0]+r2[0])
            dC3dt[0] = - duC3dz[0] - m_ad[0]*dq3dt[0] + m_cat[0]*(-r1[0]+r2[0])
            dC4dt[0] = - duC4dz[0] - m_ad[0]*dq4dt[0] + m_cat[0]*(+r1[0]-r2[0])

            dC1dt[-1] = - duC1dz[-1] - m_ad[-1]*dq1dt[-1] + m_cat[-1]*(+r1[-1]-r2[-1])
            dC2dt[-1] = - duC2dz[-1] - m_ad[-1]*dq2dt[-1] + m_cat[-1]*(-r1[-1]+r2[-1])
            dC3dt[-1] = - duC3dz[-1] - m_ad[-1]*dq3dt[-1] + m_cat[-1]*(-r1[-1]+r2[-1])
            dC4dt[-1] = - duC4dz[-1] - m_ad[-1]*dq4dt[-1] + m_cat[-1]*(+r1[-1]-r2[-1])

            dydt = np.concatenate([dC1dt,dC2dt, dC3dt, dC4dt, 
                                dq1dt, dq2dt, dq3dt,dq4dt])
            return dydt
        
        y_res = odeint(model_col, y0, t_span)
        if self.forward_flow:        
            self.y_res = y_res
        else:
            y_res_list = []
            
            for ii in range(4):
                C_tmp = y_res[ :, ii*self.N : (ii+1)*self.N ]@self.Mat_rev
                y_res_list.append(C_tmp)
            for ii in range(4,8):
                q_tmp = y_res[ :, ii*self.N : (ii+1)*self.N ]@self.Mat_rev
                y_res_list.append(q_tmp)

            y_res_list = [np.array(arr, dtype=np.float32) for arr in y_res_list] #float32로 수정
            y_res = np.concatenate(y_res_list, axis=1)
            self.y_res = y_res

        self.t_span = t_span

        C_list = []
        q_list = []
        for ii in range(4):
            C_list.append(y_res[:, ii*self.N:(ii+1)*self.N])
        for ii in range(4, 8):
            q_list.append(y_res[:, ii*self.N:(ii+1)*self.N])
        self.C_res = C_list
        self.q_res = q_list
        # Results in dictionary ? No !

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # =-=-=- Next Initial  -=-=-=-=-=
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def next_init(self, Cq_return = False, change_init = True):
        C_init = [[],]*4
        q_init = [[],]*4
        for ii in range(4):
            C_init[ii] = self.C_res[ii][-1,:]
            q_init[ii] = self.q_res[ii][-1,:]
        if change_init:
            self.C_init = C_init
            self.q_init = q_init
        if Cq_return:
            return C_init, q_init
        

    def next_init_p(self, T_feed_in, T_feed_out, Cq_return = False, change_init = True):
        C_init = [[],]*4
        q_init = [[],]*4
        for ii in range(4):
            C_init[ii] = T_feed_in/T_feed_out*self.C_res[ii][-1,:]
            q_init[ii] = self.q_res[ii][-1,:]
        if change_init:
            self.C_init = C_init
            self.q_init = q_init
        if Cq_return:
            return C_init, q_init
        
    #Define the Rep feed conditions
    def feed_condi_rep(self, y_feed_arr, P_feed, T_feed, ):
        self.y_feed1 = y_feed_arr[0]
        self.y_feed2 = y_feed_arr[1]
        self.y_feed3 = y_feed_arr[2]
        self.y_feed4 = y_feed_arr[3]

        C1_feed = self.y_feed1*P_feed*1E5/R_gas/T_feed
        C2_feed = self.y_feed2*P_feed*1E5/R_gas/T_feed
        C3_feed = self.y_feed3*P_feed*1E5/R_gas/T_feed
        C4_feed = self.y_feed4*P_feed*1E5/R_gas/T_feed
        self.C_feed_rep = [C1_feed, C2_feed, C3_feed, C4_feed,]

    def next_init_p2(self, T_feed_in, T_feed_out, Cq_return = False, change_init = True):
        C_init = [[],]*4
        q_init = [[],]*4
        for ii in range(4):
            C_factor = (T_feed_in/T_feed_out-1)*(self.C_res[0][-1,:] + self.C_res[1][-1,:] + self.C_res[2][-1,:] + self.C_res[3][-1,:])/(self.C_feed_rep[0] + self.C_feed_rep[1] + self.C_feed_rep[2] + self.C_feed_rep[3])
            C_init[0] = self.C_res[0][-1,:] + self.C_feed_rep[0]*C_factor
            C_init[1] = self.C_res[1][-1,:] + self.C_feed_rep[1]*C_factor
            C_init[2] = self.C_res[2][-1,:] + self.C_feed_rep[2]*C_factor
            C_init[3] = self.C_res[3][-1,:] + self.C_feed_rep[3]*C_factor
            q_init[ii] = self.q_res[ii][-1,:]
        if change_init:
            self.C_init = C_init
            self.q_init = q_init
        if Cq_return:
            return C_init, q_init


    def next_init_p2_test(self, y_feed_list, T_feed_in, T_feed_out, Cq_return = False, change_init = True):
        C_init = [[],]*4
        q_init = [[],]*4
        for ii in range(4):
            C_factor = (T_feed_in/T_feed_out-1)*(self.C_res[0][-1,:] + self.C_res[1][-1,:] + self.C_res[2][-1,:] + self.C_res[3][-1,:])/(y_feed_list[0]*self.C_res[0][-1,:] + y_feed_list[1]*self.C_res[1][-1,:] + y_feed_list[2]*self.C_res[2][-1,:] + y_feed_list[3]*self.C_res[3][-1,:])
            C_init[0] = self.C_res[0][-1,:]*(1 + y_feed_list[0]*C_factor)
            C_init[1] = self.C_res[1][-1,:]*(1 + y_feed_list[1]*C_factor)
            C_init[2] = self.C_res[2][-1,:]*(1 + y_feed_list[2]*C_factor)
            C_init[3] = self.C_res[3][-1,:]*(1 + y_feed_list[3]*C_factor)
            q_init[ii] = self.q_res[ii][-1,:]
        if change_init:
            self.C_init = C_init
            self.q_init = q_init
        if Cq_return:
            return C_init, q_init


    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # =-=-=- Graph drawing -=-=-=-=-=
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    # Graph: still cuts for every "t_frame" of "t_span"
    def graph_still(self, y_index, t_frame, label, filename, 
                    bbox_pos=[1.42, 0.92], figsize = None, 
                    y_limits=None, show = False,):
        t_index = range(0, len(self.t_span), t_frame)

        plt.figure(tight_layout = True, figsize = figsize)
        cc = 0
        ls_list = ['-','--','-.',':']
        if y_index == 'P':
            C_res = self.C_res
            C_ov = C_res[0] + C_res[1] + C_res[2] +C_res[3]
            P_ov = C_ov*self.T_feed*R_gas/1E5
            y_samp =P_ov
        else:
            y_samp = self.y_res[: , y_index*self.N : (y_index+1)*self.N]

        for ii, tt in zip(t_index, self.t_span[t_index]):
            C_samp = y_samp[ii,:]
            plt.plot(self.z, C_samp, 'k',
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
        if y_limits:
            plt.ylim(y_limits)
        plt.grid(ls = '--')
        plt.savefig(filename, dpi = 150, bbox_inches = 'tight')
        if show:
            plt.show()
        else:
            plt.close()

    def graph_still_mol(self, y_index, t_frame, label, filename, 
                        bbox_pos=[1.42, 0.92], figsize = None, 
                        y_limits=None, show = False,):
        t_index = range(0, len(self.t_span), t_frame)
        
        plt.figure(tight_layout = True, figsize = figsize)
        
        # Overall gas concentration
        C_ov = np.zeros_like(self.C_res[0])
        for CC in self.C_res:
            C_ov += CC
        # Select index
        if y_index == 'P':
            P_ov = C_ov*self.T_feed*R_gas/1E5
            y_samp =P_ov
        else:
            y_samp = self.C_res[y_index]/C_ov
        # For loop
        cc = 0
        ls_list = ['-','--','-.',':']
        for ii, tt in zip(t_index, self.t_span[t_index]):
            C_samp = y_samp[ii,:]
            plt.plot(self.z, C_samp, 'k',
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
        if y_limits:
            plt.ylim(y_limits)
        plt.grid(ls = '--')
        plt.savefig(filename, dpi = 150, bbox_inches = 'tight')
        if show:
            plt.show()
        else:
            plt.close()


    def graph_timelapse(self, y_index, t_frame, 
                        label, filename, figsize=None,
                        bbox_pos=[1.42, 0.92], 
                        y_limits=None, interval=100):
        fig, ax = plt.subplots(figsize=figsize, dpi = 300)
        line, = ax.plot([], [], 'k-', lw=2)
        ax.set_xlabel('z-axis (m)', fontsize=13)
        ax.set_ylabel(label, fontsize=13)
        ax.grid(ls='--')
        ax.set_xlim([(2*self.z[0]-self.z[1]), (2*self.z[-1]-self.z[-2])])
        if y_limits:
            ax.set_ylim(y_limits)

        index = np.arange(0, len(self.t_span), t_frame)
        t_span = self.t_span[index]

        # Ensure y_targ aligns with the indices
        # Part 1
        if y_index == 'P':
            C_ov = self.y_res[:, 0:self.N] + 0
            for ii in range(1,4):
                C_tmp = self.y_res[:, ii*self.N:(ii+1)*self.N] + 0
                C_ov = C_ov + C_tmp
            y_targ = C_ov * R_gas * self.T_feed / 1E5
        else:
            y_targ = self.y_res[:, y_index*self.N:(y_index+1)*self.N]
        

        # Ensure `t_span` length matches frames
        def update(frame_idx):
            tt = t_span[frame_idx]
            C_samp = y_targ[index[frame_idx], :]
            line.set_data(self.z, C_samp)
            ax.set_title(f'Profile at t = {tt:.2f} sec', fontsize=14)
            return line,

        def init():
            line.set_data([], [])
            return line,

        ani = animation.FuncAnimation(fig, update, 
                                    frames=len(index), 
                                    init_func=init, 
                                    blit=True, interval=interval)

        ani.save(filename, writer='Pillow', fps=1000/interval)
        plt.close(fig)


    def graph_timelapse_y(self, y_index, t_frame, 
                        label, filename, figsize=None,
                        bbox_pos=[1.42, 0.92], 
                        y_limits=None, interval=100):
        fig, ax = plt.subplots(figsize=figsize, dpi = 300)
        line, = ax.plot([], [], 'k-', lw=2)
        ax.set_xlabel('z-axis (m)', fontsize=13)
        ax.set_ylabel(label, fontsize=13)
        ax.grid(ls='--')
        ax.set_xlim([(2*self.z[0]-self.z[1]), (2*self.z[-1]-self.z[-2])])
        if y_limits:
            ax.set_ylim(y_limits)

        index = np.arange(0, len(self.t_span), t_frame)
        t_span = self.t_span[index]

        # Ensure y_targ aligns with the indices
        # Part 1
        C_res = self.C_res
        C_ov = C_res[0] + C_res[1]+C_res[2]+C_res[3]

        if y_index == 'P':
            y_targ = C_ov * R_gas * self.T_feed / 1E5
        else:
            y_targ = C_res[y_index]/C_ov

        # Ensure `t_span` length matches frames
        def update(frame_idx):
            tt = t_span[frame_idx]
            C_samp = y_targ[index[frame_idx], :]
            line.set_data(self.z, C_samp)
            ax.set_title(f'Profile at t = {tt:.2f} sec', fontsize=14)
            return line,

        def init():
            line.set_data([], [])
            return line,

        ani = animation.FuncAnimation(fig, update, 
                                    frames=len(index), 
                                    init_func=init, 
                                    blit=True, interval=interval)

        ani.save(filename, writer='Pillow', fps=1000/interval)
        plt.close(fig)


    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # =-=-=- Display info =-=-=-=-=-=
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def display_pac_info(self):
        """
        Display packing information.
        """
        if self.is_pac_info:
            return (
                f"Packing Information:\n"
                f"  Void Fraction (ε): {self.epsi}\n"
                f"  Catalyst Fraction (x_cat): {self.x_cat}\n"
                f"  Particle Diameter (d_p): {self.d_particle} m"
            )
        else:
            return "Packing information has not been provided yet."

    def display_ads_info(self):
        """
        Display adsorbent information.
        """
        if self.is_ads_info:
            return (
                f"Adsorbent Information:\n"
                f"  Adsorbent Density (ρ_ads): {self.rho_ads} kg/m³\n"
                f"  Mass Transfer Coefficients (k_list): {self.k_list}\n"
                f"  Isotherm Model: Defined"
            )
        else:
            return "Adsorbent information has not been provided yet."

    def display_cat_info(self):
        """
        Display catalyst reaction information.
        """
        if self.is_cat_info:
            return (
                f"Catalyst Reaction Information:\n"
                f"  Reference Forward Reaction Rate (k_f_ref): {self.k_f_ref}\n"
                f"  Reference Temperature (T_ref): {self.T_ref} K\n"
                f"  Activation Energy (E_a_f): {self.E_a_f} J/mol\n"
                f"  Catalyst Density (ρ_cat): {self.rho_cat} kg/m³\n"
                f"  Reaction Orders: {self.orders}"
            )
        else:
            return "Catalyst reaction information has not been provided yet."

    def display_feed_condi(self):
        """
        Display feed condition information.
        """
        if self.is_feed:
            return (
                f"Feed Conditions:\n"
                f"  Mole Fractions (y_feed): {self.y_feed}\n"
                f"  Feed Pressure (P_feed): {self.P_feed} bar\n"
                f"  Feed Temperature (T_feed): {self.T_feed} K\n"
                f"  Feed Velocity (u_feed): {self.u_feed} m/s\n"
                f"  Valve Constant (Cv_in): {self.Cv_in if self.Cv_in else 'Not provided'}"
            )
        else:
            return "Feed condition information has not been provided yet."

    def display_outlet_condi(self):
        """
        Display outlet condition information.
        """
        if self.is_outlet:
            return (
                f"Outlet Conditions:\n"
                f"  Mole Fractions (y_end): {self.y_end}\n"
                f"  Outlet Pressure (P_end): {self.P_end} bar\n"
                f"  Outlet Temperature (T_end): {self.T_end} K\n"
                f"  Outlet Valve Constant (Cv_end): {self.Cv_end}"
            )
        else:
            return "Outlet condition information has not been provided yet."

    def display_init_condi(self):
        """
        Display initial condition information.
        """
        if self.is_init:
            return (
                f"Initial Conditions:\n"
                f"  Initial Concentrations (C_init): {self.C_init}\n"
                f"  Initial Adsorbed Amounts (q_init): {self.q_init}\n"
                f"  Initial Temperature (T_init): {self.T_init} K"
            )
        else:
            return "Initial condition information has not been provided yet."

    def display_feed_info(self):
        """
        Display feed-related details (additional to feed conditions).
        """
        if self.is_feed:
            return (
                f"Feed Information:\n"
                f"  Feed Mole Fractions (y_feed): {self.y_feed}\n"
                f"  Feed Velocity (u_feed): {self.u_feed} m/s\n"
                f"  Feed Temperature (T_feed): {self.T_feed} K\n"
                f"  Feed Pressure (P_feed): {self.P_feed} bar\n"
                f"  Feed Concentrations (C_feed): {self.C_feed}"
            )
        else:
            return "Feed information has not been provided yet."

    def __str__(self):
        str_to_print = ""
        str_to_print+= "pac_info: " + str(self.is_pac_info) + '\n'
        str_to_print+= "ads_info: " + str(self.is_ads_info) + '\n'
        str_to_print+= "cat_info: " + str(self.is_cat_info) + '\n'
        str_to_print+= "feed_condi: " + str(self.is_feed) + '\n'
        str_to_print+= "outlet_condi: " + str(self.is_outlet) + '\n'
        str_to_print+= "init_condi: " + str(self.is_init) + '\n'

        return str_to_print
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # =-=- Pressure Setting =-=-=-=-=
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def set_Cv_Pfeed(self, P_mid, x_vel_feed_out=0.8):
        if not hasattr(self, 'u_feed'):
            print("Feed velocity (u_feed) is not defined.")
            return
        if not hasattr(self, 'P_end'):
            print("Outlet pressure (P_end) is not defined.")
            return
        if not hasattr(self, 'epsi'):
            print("Void fractio (epsi) is not defined.")
            return
        if not hasattr(self, 'C_feed'):
            print("Feed concentration (C_feed) is not defined.")
            return
        if not hasattr(self, 'T_feed'):
            print("Feed temperature (CT_feed) is not defined.")
            return
        u_out_assum = x_vel_feed_out*self.u_feed
        
        ep = self.epsi

        y_feed_arr = np.array(self.C_feed)/np.sum(self.C_feed)
        C_ov_mid = P_mid*1E5/R_gas/self.T_feed
        rho_g_assum = np.sum(np.array(Mw_gas)*y_feed_arr)*C_ov_mid # (kg/m^3)

        term1 = 150*(1-ep)**2/ep**2*mu_gas/self.d_particle**2*u_out_assum
        term2 = 1.75*(1-ep)/ep*rho_g_assum/self.d_particle*u_out_assum**2
        Del_P = (term1 + term2)*self.L
        P_bed_end = P_mid - Del_P/2/1E5
        Cv_out_set = u_out_assum/(P_bed_end - self.P_end)
        P_bed_begin = P_mid + Del_P/2/1E5
        success = True

        '''
        print('T_feed: ', self.T_feed)
        print('P_mid: ', P_mid)
        print('R_gas: ', R_gas)
        print('C_feed: ', self.C_feed)
        print('y_feed_arr: ', y_feed_arr)
        print('C_ov_mid: ', C_ov_mid)
        print('Del_P: ', Del_P/1E5, ' bar')
        print('P_bed_end:', P_bed_end)
        print('P_end:', self.P_end)
        '''

        if P_bed_end < self.P_end:
            success = False
            print('Target P_mid is too low for given P_end!')
            return Cv_out_set, P_bed_begin, success
        else:
            return Cv_out_set, P_bed_begin, success

        # ...additional logic for setting Cv...


# %%
# Test SEWGS class

if __name__ == '__main__':
    L = 2
    N = 41
    epsi = 0.4
    x_cat = 0.7
    d_p = 2E-3
    ## 1) Define AdsCatCol
    acc1 = AdsCatColumn(L, N,)
    ## 2) Sample pac_info: packing info
    acc1.pac_info(epsi, x_cat, d_p)
    print(acc1)
    ## 3) Sample ads_info: isotherm func & density
    rho_ads = 1000 # kg/m3  //density of adsorbent
    k_list = [0.02,]*4
    qm1 = 0.02 # H2
    qm2 = 0.3 # CO
    qm3 = 0.6 # H2O
    qm4 = 3.0 # CO2

    b1 = 0.05
    b2 = 0.1
    b3 = 0.3
    b4 = 0.5
    def iso(P1, P2, P3, P4, T): # isotherm function
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
    acc1.ads_info(iso, k_list, rho_ads, )
    print(acc1)

    ## 4) Sample cat_info: rate constant func & density
    rho_cat = 1000 # kg/m3  //density of catalyst
    k_f_ref = 1E-3 # m/s
    T_ref = 623 # K
    E_a_f = 1E4 # J/mol
    acc1.cat_info(k_f_ref, T_ref, E_a_f, rho_cat, )
    print(acc1)

    ## 5) Sample feed_condi: feed composition & conditions
    y_feed_list = [0, 0.5, 0.5, 0]
    P_feed = 7.0 # bar
    T_feed = 623 # K
    u_feed = 0.05 # m/s
    acc1.feed_condi_y(y_feed_list, P_feed, T_feed, u_feed, )
    #acc1.feed_condi_y()
    print(acc1)
    
    ## 6) Sample outlet_condi: exit composition & conditions
    y_end_list = [1, 0, 0, 0.0]
    P_end = 5.5 # bar
    Cv_end = 2E-2
    T_end = 623 # K
    acc1.outlet_condi_y(y_end_list, P_end, T_end, Cv_end, )
    print(acc1)
    
    ## 7) Sample init_condi: initial composition & conditions
    P_init = 5.7*np.ones(N)
    T_init = 623*np.ones(N)
    y1_i = 1.0*np.ones([N,])
    y2_i = 0.0*np.ones([N,])
    y3_i = 0.0*np.ones([N,])
    y4_i = 0.0*np.ones([N,])
    y_init = [y1_i, y2_i, y3_i, y4_i,]
    acc1.init_condi_y(P_init,y_init, None, T_init, True)
    ## 8) Flow direction setting
    acc1.Flow_direction(False)
    print(acc1)
    
    Cv_end_cal, P_feed_cal, succ = acc1.set_Cv_Pfeed(P_mid = 7.0,
                                                     x_vel_feed_out = 0.6,)
    print('Cv out_cal: ', Cv_end_cal)
    print('P_feed: ', P_feed_cal)
    acc1.feed_condi_y(y_feed_list, P_feed_cal,
                      T_feed, u_feed,)
    acc1.outlet_condi_y(y_end_list, P_end, T_end, 
                        Cv_end_cal,)
    
# %%
# Check the information
# & RUN the simulations
# with display methods 
# %%
if __name__ == '__main__':
    '''
    ## 8) Display the information
    print(acc1.display_pac_info())
    print(acc1.display_ads_info())
    print(acc1.display_cat_info())
    print(acc1.display_feed_condi())
    print(acc1.display_outlet_condi())
    print(acc1.display_init_condi())
    print(acc1.display_feed_info())
    '''
    ## 9) Run simulations
    t_ran = np.arange(0,320+0.0025, 0.0025)
    acc1.run_mamo(t_ran,)

# %%
# Graph drawing
# %%
if __name__ == '__main__':
    import os
    os.makedirs('test_res', exist_ok = True)
    figsize_test = [12.5,5]
    ShowGraph = False
    acc1.graph_still(0, t_frame = 5000, 
                     label = 'H$_{2}$ concentration (mol/m$_{3}$)',
                     filename='test_res/sim_c01_rxn_C1.png',
                     figsize = figsize_test,
                     show = ShowGraph)
    
    acc1.graph_still(4, t_frame = 5000, 
                     label = 'H$_{2}$ uptake (mol/kg)',
                     filename='test_res/sim_c01_rxn_q1.png',
                     figsize = figsize_test,
                     show = ShowGraph)
    
    acc1.graph_still(5, t_frame = 5000, 
                     label = 'CO uptake (mol/kg)',
                     filename='test_res/sim_c01_rxn_q2.png',
                     figsize = figsize_test,
                     show = ShowGraph)
    
    acc1.graph_still(6, t_frame = 5000, 
                     label = 'H$_{2}$O uptake (mol/kg)',
                     filename='test_res/sim_c01_rxn_q3.png',
                     figsize = figsize_test,
                     show = ShowGraph)

    acc1.graph_still('P', t_frame = 5000,
                     label = 'Pressure (bar)',
                     filename='test_res/sim_c01_rxn_P1.png',
                     figsize = figsize_test,
                     show = ShowGraph)
    acc1.graph_timelapse('P',t_frame=2000,label='P (bar)',
                         filename ='test_res/sim_c01_rxn_P.gif',
                         y_limits=[4.5,7.4], interval = 100)
    
    # mole fraction grahp?
    ShowGraph_mol = True
    acc1.graph_still('P', t_frame = 5000, 
                     label = 'Pressure (bar)',
                     filename='test_res/sim_c01_rxn_P.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)
    
    acc1.graph_still_mol(0, t_frame = 5000,
                     label = 'H$_{2}$ mole frac.',
                     filename='test_res/sim_c01_rxn_y1.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)
    
    acc1.graph_still_mol(1, t_frame = 5000,
                     label = 'CO mole frac.',
                     filename='test_res/sim_c01_rxn_y2.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)
    
    acc1.graph_still_mol(2, t_frame = 5000,
                     label = 'H$_{2}$O mole frac.',
                     filename='test_res/sim_c01_rxn_y3.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)
    
    acc1.graph_still_mol(3, t_frame = 5000,
                     label = 'CO$_{2}$ mole frac.',
                     filename='test_res/sim_c01_rxn_y4.png',
                     figsize = figsize_test,
                     show = ShowGraph_mol)
    
    


# %%
