
# Plot speeds of different BIWI clusters obtained by running run_on_host.sh
# on different hosts. 
# Author: Christian F. Baumgartner (c.f.baumgartner@gmail.com)


import matplotlib.pyplot as plt
import numpy as np

data = []
hostnames = ['biwirender04', 'biwirender08', 'biwirender06', 'bmicgpu01']

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Comparison of clusters')

ax.set_xlabel('matrix size')
ax.set_ylabel('execution time (secs)')

for hostname in hostnames:

    npzfile = np.load('%s.npz' % hostname)

    sizes = npzfile['arr_0']
    times = npzfile['arr_1']

    ax.plot(sizes, times, label=hostname)

plt.legend()
plt.show()
