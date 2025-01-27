import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5, 51)
y = np.exp(-x**2)
#plt.figure()
plt.subplots(figsize = [8,2])
plt.plot(x,y,label = 'label test test')

plt.grid(linestyle = ':')
plt.legend(fontsize = 14,loc = 'upper center', bbox_to_anchor = [1.42, 0.92])

plt.show()
