import sys
sys.path.append('./scripts')
import topovis
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import h5py
import numpy as np
import pathlib

figs_dir = './figs/compare_interpolation/'
ext = '.png'
DPI = 200

def figpath(filename):
    return figs_dir+filename+ext

args_list = [
    {'geom': 'circ', 'ns': '32', 'fs': '1', 'fx': '1'},
    {'geom': 'circ', 'ns': '32', 'fs': '4', 'fx': '1', 'method': 'rbfi'},
    {'geom': 'circ', 'ns': '32', 'fs': '4', 'fx': '1', 'method': 'rgi'},
    {'geom': 'circ', 'ns': '128', 'fs': '1', 'fx': '1'},
]

results : list[topovis.ToPoVisData] = []
names = []

vmin = np.inf
vmax = -np.inf

for args in args_list:
    data_dir = './data/'+args['geom']+'/ns'+args['ns']+'/'
    in_path = data_dir+'gkwdata.h5'

    if 'method' in args:
        res = topovis.main(['-vv', '--omit-axes', '--periodic', '--interpolator', args['method'], in_path, args['fx'], args['fs']])
    else:
        res = topovis.main(['-vv', '--omit-axes', in_path, args['fx'], args['fs']])

    results.append(res)

    min = float(np.min(res.pot))
    max = float(np.max(res.pot))
    if min < vmin: vmin = min
    if max > vmax: vmax = max

def plot(res):
    kwargs = {'vmin': vmin, 'vmax': vmax}

    fig, ax = plt.subplots()
    fig.tight_layout()

    res.plot(fig, ax, **kwargs)

    path = figs_dir + 'circ/' + str(res.interpolator) + 'ns' + str(res.ns) + '-fs' + str(res.fs) + ext
    plt.savefig(path, dpi=DPI)

# plots diffs

exact = results[3]
rbfi = results[2]
rgi = results[1]

diff_rgi = np.abs(rgi.pot - exact.pot)
vmin_rgi = np.min(diff_rgi)
vmax_rgi = np.max(diff_rgi)

diff_rbfi = np.abs(rbfi.pot - exact.pot)
vmin_rbfi = np.min(diff_rbfi)
vmax_rbfi = np.max(diff_rbfi)

vmin = np.min([vmin_rbfi, vmin_rgi])
vmax = np.max([vmax_rbfi, vmax_rgi])

kwargs = {'cmap':'Reds','vmin': vmin, 'vmax': vmax}

fig, ax = plt.subplots()
fig.tight_layout()

topovis.plot(rgi.r, rgi.z, diff_rgi, fig, ax, omit_axes=True, **kwargs)

path = figs_dir + 'circ/rgi/ns' + str(rgi.ns) + '-fs' + str(rgi.fs) + '-diff' + ext
plt.savefig(path, dpi=DPI)

fig, ax = plt.subplots()
fig.tight_layout()
topovis.plot(rbfi.r, rbfi.z, diff_rbfi, omit_axes=True, **kwargs)

path = figs_dir + 'circ/rbfi/ns' + str(rbfi.ns) + '-fs' + str(rbfi.fs) + '-diff' + ext
plt.savefig(path, dpi=DPI)