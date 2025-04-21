import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./scripts')
import topovis

figs_dir = 'figs/triangulation/regular_grid/'
plt.rcParams.update({'font.size': 14})

x = np.arange(4)
y = np.arange(3)

xy = xx, yy = np.meshgrid(x, y, indexing='ij')

triangles = topovis.make_regular_triangles(len(x), len(y))

fig, ax = plt.subplots()

ax.scatter(xx, yy, color='0', zorder=5)
ax.triplot(np.ravel(xx),np.ravel(yy),triangles, color='lightcoral', lw=2.5)

ax.invert_yaxis()

ax.set_xticks(x)
ax.set_yticks(y)
ax.set_xticklabels(list(map(str, x)), )
ax.set_yticklabels(list(map(str, y)))
ax.xaxis.tick_top()
ax.set_xlabel('i')
ax.xaxis.set_label_position('top') 
ax.set_ylabel('j')


plt.savefig(figs_dir+'plot.svg')
