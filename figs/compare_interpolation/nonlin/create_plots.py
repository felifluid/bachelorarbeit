import sys
sys.path.append('./scripts')
import topovis
import matplotlib.pyplot as plt
import numpy as np

figs_dir = './figs/compare_interpolation/nonlin/'
ext = '.png'
DPI = 400

def figpath(filename):
    return figs_dir+filename+ext

args_list = [
    {'ns': '64', 'fs': '1', 'fx': '1', 'method': 'linear'},
    {'ns': '16', 'fs': '1', 'fx': '1', 'method': 'linear'},
    {'ns': '16', 'fs': '4', 'fx': '1', 'method': 'linear'},
]

results : list[topovis.ToPoVisData] = []
names = []

vmin = np.inf
vmax = -np.inf

for args in args_list:
    data_dir = './data/nonlin/circ/'+'ns'+args['ns']+'/'
    in_path = data_dir+'gkwdata.h5'

    res = topovis.main(['-vv', '--omit-axes', '--period', '--method', args['method'], '--triang-method', 'regular', '--legacy-gmap', in_path, args['fx'], args['fs']])

    results.append(res)

    min = float(np.min(res.pot))
    max = float(np.max(res.pot))
    if min < vmin: vmin = min
    if max > vmax: vmax = max



def plot(res):
    # kwargs = {'vmin': vmin, 'vmax': vmax}

    fig, ax = plt.subplots()
    fig.tight_layout()

    res.plot(fig, ax)

    path = figs_dir + 'ns' + str(res.ns//res.fs) + '-fs' + str(res.fs) + ext
    plt.savefig(path, dpi=DPI)

# plot results

for res in results:
    plot(res)


# plots diffs

def difference(a,b):
    return np.abs((a-b)/np.max(b))

exact = results.pop(0)
ns16_fs1 = results.pop(0)

vmin = np.inf
vmax = -np.inf

for res in results:
    diff = difference(res.pot, exact.pot)
    min = float(np.min(diff))
    max = float(np.max(diff))
    if min < vmin: vmin = min
    if max > vmax: vmax = max

    # save diff into ToPoVis Data object
    res.diff = diff

for res in results:
    kwargs = {'cmap':'Reds','vmin': vmin, 'vmax': vmax}

    fig, ax = plt.subplots()
    fig.tight_layout()

    topovis.plot(res.r, res.z, res.diff, fig, ax, omit_axes=True, **kwargs)

    path = figs_dir + 'ns' + str(res.ns) + '-fs' + str(res.fs) + '-diff' + ext
    plt.savefig(path, dpi=DPI)