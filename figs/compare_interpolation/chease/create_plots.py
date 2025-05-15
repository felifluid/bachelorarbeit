import sys
sys.path.append('./scripts')
import topovis
import matplotlib.pyplot as plt
import numpy as np

data_dir = 'data/lin/chease/bugfix/'
figs_dir = './figs/compare_interpolation/chease/'
ext = '.png'
DPI = 400

def figpath(filename):
    return figs_dir+filename+ext

def h5path(ns: int):
    return data_dir + 'ns' + str(ns) + '/gkwdata.h5' 

lo = topovis.main(['--omit-axes', h5path(ns=128)])
hi = topovis.main(['--omit-axes', h5path(ns=512)])
rgi = topovis.main(['--omit-axes', '--interpolator', 'rgi', '--fs', '4', '--method', 'cubic', h5path(ns=128)])
rbfi = topovis.main(['--omit-axes', '--interpolator', 'rbfi', '--fs', '4', '--method', 'cubic', h5path(ns=128)])

results = [lo, hi, rgi, rbfi]

# determine vmin & vmax

vmin = np.inf
vmax = -np.inf
for res in results:
    min = float(np.min(res.pot))
    max = float(np.max(res.pot))
    if min < vmin: vmin = min
    if max > vmax: vmax = max

print("Creating Plot")

fig, axs = plt.subplots(2,2, figsize=(9,7.5))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)

kwargs = {'vmin': vmin, 'vmax': vmax, 'vcenter': 0, 'omit_axes': True, 'omit_cbar': False}

xlim = (1.245, 1.26)
ylim = (-0.02, 0.092)

rgi.plot(fig, axs[0,0], **kwargs)
rbfi.plot(fig, axs[1,0], **kwargs)

# plots diffs
def difference(a,b):
    return np.abs((a-b)/np.max(b))

print('Calculating diffs')

rgi_diff = difference(hi.pot, rgi.pot)
print(f'RGI: max {np.max(rgi_diff)}, mean {np.mean(rgi_diff)}')
rbfi_diff = difference(hi.pot, rbfi.pot)
print(f'RGI: max {np.max(rbfi_diff)}, mean {np.mean(rbfi_diff)}')

vmin_diff = np.min([rgi_diff, rbfi_diff])
vmax_diff = np.max([rgi_diff, rbfi_diff])


print("Plotting differences")

kwargs = {'vmin': vmin_diff, 'vmax': vmax_diff, 'vcenter': None, 'omit_axes': True, 'cmap': 'Reds'}

kwargs['pot'] = rgi_diff
rgi.plot(fig, axs[0,1], **kwargs)

kwargs['pot'] = difference(hi.pot, rbfi.pot)
rbfi.plot(fig, axs[1,1], **kwargs)

cols = ['upscaled potential', 'relative difference']
rows = ['RGI', 'RBFI']

for ax, col in zip(axs[0], cols):
    ax.set_title(col)

for ax, row in zip(axs[:, 0], rows):
    ax.set_ylabel(row, size='large')

for row in axs:
    for ax in row:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('auto')

fig.savefig(figs_dir + 'interpolation.png', dpi=DPI)

kwargs = {'vmin': vmin, 'vmax': vmax, 'vcenter': 0, 'omit_axes': True, 'omit_cbar': False}


fig, ax = plt.subplots()

lo.plot(fig, ax, **kwargs)

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('auto')

fig.savefig(figs_dir + 'ns126.png', dpi=DPI)

fig, ax = plt.subplots()

hi.plot(fig, ax, **kwargs)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('auto')

fig.savefig(figs_dir + 'ns512.png', dpi=DPI)

print("Done")