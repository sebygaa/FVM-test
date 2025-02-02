# %% 
# Importing os
# %%
import os
import numpy as np
import time
import psutil
def cpucheck():
    cpu_use_list =[]
    for ii in range(4):
        cpu_use_test = psutil.cpu_percent(0.5)
        cpu_use_list.append(cpu_use_test)
    arg_min = np.argmin(cpu_use_list)
    cpu_dum = cpu_use_list.pop(arg_min)
    cpu_perc_average = np.mean(cpu_dum)
    return cpu_perc_average

# %%
# Temperature variables
# %%
#P_arr = np.arange(3,21,1)
#T_arr = np.arange(300,601,50)+273
P_arr = np.array([3,20])
T_arr = np.array([573,873])
#print(P_arr)
#print(T_arr)
fol_nam= 'PDE_res'

os.makedirs(fol_nam, exist_ok=True)
#os.chdir(fol_nam)

for pp in P_arr:
    for tt in T_arr:
        run_comm = 'python '+'SEWGS_argv.py '+ f'{tt} {pp}'
        os.system(run_comm)
        print(run_comm)
        tic = time.time()
        
        cpu_perc = cpucheck()
        for ii in range(100000):
            if cpu_perc > 85:
                print('CPU usage = ')
                print(cpu_perc)
                time.sleep(10)
            else:
                #print(cpu_perc)
                toc =time.time() - tic
                print('CPU time = ')
                print(f' {toc/60:.2f} min')
                break

for pp in P_arr:
    for tt in T_arr:
        f_nam1 = f'PDE_C_data_T{tt:03.0f}K_P{pp:02.0f}bar.csv'
        f_nam2 = f'PDE_q_data_T{tt:03.0f}K_P{pp:02.0f}bar.csv'
        os.system(f'mv {f_nam1} ./{fol_nam}/')
        os.system(f'mv {f_nam2} ./{fol_nam}/')

#T_list = []
#P_list = []