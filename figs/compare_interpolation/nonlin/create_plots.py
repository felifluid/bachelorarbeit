import sys
sys.path.append('./scripts')
import topovis
import matplotlib.pyplot as plt
import numpy as np

figs_dir = './figs/compare_interpolation/nonlin/'
data_dir = './data/nonlin/circ/sophia/'
ext = '.png'
DPI = 400

def figpath(filename):
    return figs_dir+filename+ext

args_list = [
    {'dsf': '0', 'fs': '1'},
    {'dsf': '2', 'fs': '1'},
    {'dsf': '4', 'fs': '1'},
    {'dsf': '8', 'fs': '1'},
    {'dsf': '2', 'fs': '2'},
    {'dsf': '4', 'fs': '4'},
    {'dsf': '8', 'fs': '8'},
]

intrp_idx = 4

def args_to_name(args: dict[str, str]):
    return 'dsf' + args['dsf'] + '-fs' + args['fs']

results : list[topovis.ToPoVisData] = []
names = []

vmin = np.inf
vmax = -np.inf

print("Evaluating poloidal cross-sections")

for idx, args in enumerate(args_list):
    in_path = data_dir+'gkwdata.h5'

    print(f'Evaluating dataset {idx}')
    res = topovis.main(['--omit-axes', '--method', 'cubic', '--triang-method', 'regular', '--dsf', args['dsf'],  '--legacy-gmap', in_path, '1', args['fs']])

    results.append(res)

    min = float(np.min(res.pot))
    max = float(np.max(res.pot))
    if min < vmin: vmin = min
    if max > vmax: vmax = max

def plot(res):
    kwargs = {'vmin': vmin, 'vmax': vmax, 'vcenter': 0}

    fig, ax = plt.subplots()
    fig.tight_layout()

    res.plot(fig, ax, **kwargs)

    return fig, ax

# plot results

print("Plotting datasets")

for idx, res in enumerate(results):
    fig, ax = plot(res)
    path = figs_dir + args_to_name(args_list[idx]) + ext
    plt.savefig(path, dpi=DPI)


# plots diffs

def difference(a,b):
    return np.abs((a-b)/np.max(b))

exact = results[0]

print("Calculating differences")

vmin = np.inf
vmax = -np.inf

for idx, res in enumerate(results[intrp_idx:], intrp_idx):
    diff = difference(res.pot, exact.pot)
    min = float(np.min(diff))
    max = float(np.max(diff))
    if min < vmin: vmin = min
    if max > vmax: vmax = max

    # save diff into ToPoVis Data object
    res.diff = diff


print("Plotting differences")

for idx, res in enumerate(results[intrp_idx:], intrp_idx):
    # explicitely setting vcenter to None to disable centering cbar
    kwargs = {'cmap':'Reds','vmin': vmin, 'vmax': vmax, 'vcenter': None}

    fig, ax = plt.subplots()
    fig.tight_layout()

    topovis.plot(exact.r, exact.z, res.diff, fig, ax, omit_axes=True, **kwargs)

    path = figs_dir + args_to_name(args_list[idx]) + '-diff' + ext
    plt.savefig(path, dpi=DPI)

print("Done")