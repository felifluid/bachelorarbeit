import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import sys
sys.path.append('./scripts')
import topovis

ns = 10
s_spacing = 1/ns
overlap = 2
s = np.linspace(-0.5+s_spacing/2, 0.5-s_spacing/2, ns)
s_e = topovis.extend_regular_array(s, overlap)

fig, ax = plt.subplots()

ax.set_xlabel("s")
ax.set_xlim(s_e[0], s_e[-1])
ax.set_xticks(s_e)

ax.set_ylim(-0.4, 1.4)
ax.set_ylabel("ζ")


kwargs = {'edgecolor': '0', 'hatch': '/'}
ax.add_patch(Rectangle((s_e[-(1+overlap)], 0), s_spacing*overlap, 1, facecolor='tab:red', alpha=0.5, **kwargs))
ax.add_patch(Rectangle((s_e[overlap], 0.4), s_spacing*overlap, 0.6, facecolor='tab:red', **kwargs))
ax.add_patch(Rectangle((s_e[overlap], 0), s_spacing*overlap, 0.4, facecolor='tab:orange', **kwargs))
ax.add_patch(Rectangle((s_e[overlap], 1), s_spacing*overlap, 0.4, facecolor='tab:orange', alpha=0.5, **kwargs))


kwargs = {'edgecolor': '0', 'hatch': '.'}

ax.add_patch(Rectangle((s_e[0], 0), s_spacing*overlap, 1, facecolor='tab:blue', alpha=0.5, **kwargs))
ax.add_patch(Rectangle((s_e[-(1+overlap)-overlap], 0), s_spacing*overlap, 0.6, facecolor='tab:blue', alpha=1, **kwargs))
ax.add_patch(Rectangle((s_e[-(1+overlap)-overlap], 0.6), s_spacing*overlap, 0.4, facecolor='tab:cyan', alpha=1, **kwargs))
ax.add_patch(Rectangle((s_e[-(1+overlap)-overlap], -0.4), s_spacing*overlap, 0.4, facecolor='tab:cyan', alpha=0.5, **kwargs))

ax.axhline(1, color='0', linestyle='dashed')
ax.axhline(0, color='0', linestyle='dashed')

ax.axvline(s_e[overlap], color='0', linestyle='dashed')
ax.axvline(s_e[-overlap-1], color='0', linestyle='dashed')

plt.savefig('figs/boundary_conditions/linear.svg')