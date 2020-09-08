import matplotlib.pyplot as plt
import numpy as np

delta = [99, 50, 10, 1, 0.1, 0.01]
delta = np.array(delta)/100
success = [53, 97.8, 99.1, 99.5, 100, 100]
time = [0.48, 2.54, 3.53, 3.71, 3.84, 4.07]
time = np.array(time) + 4.8

fig, ax1 = plt.subplots()
ax1.plot(delta, success, 'b')
ax1.set_xlabel('Upper Risk Bound', fontsize=20)
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Success Rate [%]', color='b', fontsize=20)
ax1.tick_params('y', colors='b', labelsize='large')
ax1.tick_params('x', labelsize='large')
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(delta, time, 'r')
ax2.set_ylabel('Mean Completion Tme [s]', color='r', fontsize=20)
ax2.tick_params('y', colors='r', labelsize='large')

fig.tight_layout()
plt.xscale('log')
plt.grid()
plt.show()
