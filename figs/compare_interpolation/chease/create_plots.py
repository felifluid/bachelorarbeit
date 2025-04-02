import sys
sys.path.append('./scripts')
import topovis
import matplotlib.pyplot as plt
import numpy as np

data_dir = 'data/chease/global-aug/'
figs_dir = './figs/compare_interpolation/chease/'
ext = '.svg'
DPI = 400

def figpath(filename):
    return figs_dir+filename+ext

args_list = [
    {'geom': 'chease', 'nspacing': '8', 'fs': '1', 'fx': '1'},
    {'geom': 'chease', 'nspacing': '8', 'method': 'rgi', 'fs': '4', 'fx': '1'},
]

results : list[topovis.ToPoVisData] = []
names = []

vmin = np.inf
vmax = -np.inf

for args in args_list:
    in_path = data_dir+'nspacing'+args['nspacing']+'/gkwdata.h5'

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
    if res.fs == 1 and res.fx == 1:
        folder = 'original'
    else:
        folder = str(res.interpolator)
    path = figs_dir + folder + '/' + 'fs' + str(res.fs) + '-fx' +str(res.fx) + ext
    plt.savefig(path, dpi=DPI)

for res in results:
    plot(res)
