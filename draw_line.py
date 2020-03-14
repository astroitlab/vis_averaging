import matplotlib.pyplot as plt
import numpy as np

nch = ['2', '4', '8', '16', '32']  # 5 MS and avg_time=2

layer1 = np.array([13.2, 10.6, 9.7, 8.8, 8.1])  # non-block
layer2 = np.array([22.8, 20.4, 19.4, 19.0, 18.4])  # block

delt_t = layer2 - layer1
p = (layer1 + layer2) / 2
x_index = np.arange(len(nch))

plt.plot(x_index, layer1, marker='o', label='$t_1=0,t_2=0,t_3=0$')
plt.plot(x_index, layer2, marker='s', label='$t_1=2,t_2=5,t_3=3$')

for x in x_index:
    plt.text(x, p[x], '$\Delta t=%.1f$' % delt_t[x], ha='left', fontsize=10)
    plt.annotate('', xy=(x, layer2[x]), xytext=(x, p[x]), arrowprops=dict(arrowstyle='->'))  
    plt.annotate('', xy=(x, layer1[x]), xytext=(x, p[x]), arrowprops=dict(arrowstyle='->'))

plt.legend()
plt.xticks(x_index, nch)
plt.xlabel('Averaged channels ($nch$)')
plt.ylabel('Time (s)')

plt.savefig('Scale_nch')
