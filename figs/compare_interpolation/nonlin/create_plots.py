import sys
sys.path.append('./scripts')
import topovis
import matplotlib.pyplot as plt
import numpy as np

figs_dir = './figs/compare_interpolation/nonlin/'
data_dir = './data/nonlin/circ/sophia/'
ext = '.png'
DPI = 600

def figpath(filename):
    return figs_dir+filename+ext

exact = {'dsf' : '0', 'fs': '1'}

downscaled = [
    {'dsf': '2', 'fs': '1'},
    {'dsf': '4', 'fs': '1'},
    {'dsf': '8', 'fs': '1'},
    {'dsf': '16', 'fs': '1'},
]

upscaled = [
    {'dsf': '2', 'fs': '2'},
    {'dsf': '4', 'fs': '4'},
    {'dsf': '8', 'fs': '8'},
    {'dsf': '16', 'fs': '16'},
]

cols = ['Downsampled', 'Re-Upscaled', 'Relative Difference']
rows = ['DSF2', 'DSF4', 'DSF8', 'DFS16']

def args_to_name(args: dict[str, str]):
    return 'dsf' + args['dsf'] + '-fs' + args['fs']

upscaled_results : list[topovis.ToPoVisData] = []
downscaled_results : list[topovis.ToPoVisData] = []
names = []

vmin = np.inf
vmax = -np.inf

print("Evaluating downscaled potential")
for idx, args in enumerate(downscaled):
    in_path = data_dir+'gkwdata.h5'

    print(f'Evaluating dataset {idx}')

    res = topovis.main(['--omit-axes', '--method', 'linear', '--dsf', args['dsf'], '--legacy-gmap', in_path, '1', args['fs']])

    downscaled_results.append(res)

    min = float(np.min(res.pot))
    max = float(np.max(res.pot))
    if min < vmin: vmin = min
    if max > vmax: vmax = max

print("Evaluating upscaled potential")
for idx, args in enumerate(upscaled):
    in_path = data_dir+'gkwdata.h5'

    print(f'Evaluating dataset {idx}')

    res = topovis.main(['--omit-axes', '--method', 'cubic', '--triang-method', 'regular', '--dsf', args['dsf'],  '--legacy-gmap', in_path, '1', args['fs']])

    upscaled_results.append(res)

    min = float(np.min(res.pot))
    max = float(np.max(res.pot))
    if min < vmin: vmin = min
    if max > vmax: vmax = max

exact = topovis.main([in_path, '1', '1'])

# plot results

print("Creating Plot")

fig, axs = plt.subplots(len(upscaled),3, figsize=(9,2.5*len(upscaled)))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)

kwargs = {'vmin': vmin, 'vmax': vmax, 'vcenter': 0, 'omit_cbar': True}

for idx, res in enumerate(downscaled_results):
    res.plot(fig, axs[idx,0], **kwargs)

for idx, res in enumerate(upscaled_results):
    res.plot(fig, axs[idx,1], **kwargs)


# plots diffs

def difference(a,b):
    return np.abs((a-b)/np.max(b))

print("Calculating differences")

vmin = np.inf
vmax = -np.inf

for idx, res in enumerate(upscaled_results):
    diff = difference(res.pot, exact.pot)
    min = float(np.min(diff))
    max = float(np.max(diff))
    if min < vmin: vmin = min
    if max > vmax: vmax = max

    # save diff into ToPoVis Data object
    res.diff = diff
    print(f"Data {idx}: Max diff {np.max(diff)}, Mean diff {np.mean(diff)}")


print("Plotting differences")

for idx, res in enumerate(upscaled_results):
    # explicitely setting vcenter to None to disable centering cbar
    kwargs = {'cmap':'Reds','vmin': vmin, 'vmax': vmax, 'vcenter': None}

    topovis.plot(exact.r, exact.z, res.diff, fig, axs[idx, 2], omit_axes=True, **kwargs)


for ax, col in zip(axs[0], cols):
    ax.set_title(col)

for ax, row in zip(axs[:, 0], rows):
    ax.set_ylabel(row, size='large')

fig.savefig(figs_dir + 'plot.png', dpi=DPI)

print("Done")