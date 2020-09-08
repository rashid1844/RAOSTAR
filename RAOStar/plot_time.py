import matplotlib.pyplot as plt
import numpy as np

h = [1,2,3,4]
n_1 = [0.0020, 0.0116, 0.0401, 0.0772]
n_2 = [0.0046, 0.0484, 0.2673, 0.9911]
n_3 = [0.0181, 0.1960, 2.5658, 35.9623]
n_1 = np.array(n_1)*1000
n_2 = np.array(n_2)*1000
n_3 = np.array(n_3)*1000


fig, ax1 = plt.subplots()
ax1.plot(h, n_1, 'b', label='n=1')
ax1.plot(h, n_2, 'g', label='n=2')
ax1.plot(h, n_3, 'r', label='n=3')
ax1.set_xlabel('Planning Horizon', fontsize=20)
ax1.set_ylabel('Computational Time [ms]', fontsize=20)
ax1.tick_params('y', labelsize='large')
ax1.tick_params('x', labelsize='large')

horiz_line_data = np.array([200 for i in range(len(h))])
plt.plot(h, horiz_line_data, 'k--') 

fig.tight_layout()
plt.yscale('log')
plt.legend(prop={'size': 20})
plt.grid()
plt.show()
