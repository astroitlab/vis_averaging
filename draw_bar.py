import matplotlib.pyplot as plt
import numpy as np

n_procs = ['2', '4', '8', '12', '16', '20']  # avg_time=2, avg_channel=4

p_map = np.array([70.27, 32.4, 15.5, 11.36, 9.70, 8.70])
p_total = np.array([81.09, 45.05, 28.71, 23.48, 21.92, 19.57])
p_combine = p_total - p_map
p_saturation = np.array([6.35, 6.14, 6.62, 7.04, 7.64, p_map[-1]])

bar_width = 0.3
space = 0.2
index = np.arange(len(n_procs))
p_index = index + bar_width + space

plt.bar(p_index, p_map, width=bar_width, label='Map-Calculations', fc='#8FBC8F')
plt.bar(p_index, p_combine, width=bar_width, bottom=p_map, label='Reduce', fc='#B0C4DE')
plt.plot(p_index, p_saturation, marker='*', ls='--', label='Process Saturation')

plt.xticks(p_index, n_procs)
plt.xlabel('process quantity')
plt.ylabel('time (s)')

plt.legend()
plt.savefig('scale_process')
