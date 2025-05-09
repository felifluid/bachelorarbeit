import sys
sys.path.append('./scripts')
import topovis
import matplotlib.pyplot as plt
import numpy as np

figs_dir = './figs/compare_interpolation/lin/'
data_dir = './data/lin/circ/'
ext = '.png'
DPI = 400

def figpath(filename):
    return figs_dir+filename+ext

def args_to_name(args: dict[str, str]):
    return 'dsf' + args['dsf'] + '-fs' + args['fs']


in_path = data_dir + 'ns128/gkwdata.h5'
print('Calculating original potential')
hi = topovis.main([in_path, '1', '1'])

in_path = data_dir + 'ns32/gkwdata.h5'
print('Interpolating using RGI')
rgi = topovis.main(['--interpolator', 'rgi', in_path, '1', '4'])

print('Interpolating using RBFI')
rbfi = topovis.main(['--interpolator', 'rbfi', in_path, '1', '4'])

results = [hi, rgi, rbfi]

# determine vmin & vmax

vmin = np.inf
vmax = -np.inf
for res in results:
    min = float(np.min(res.pot))
    max = float(np.max(res.pot))
    if min < vmin: vmin = min
    if max > vmax: vmax = max


# plot results

print("Creating Plot")

fig, axs = plt.subplots(2,2, figsize=(9,7.5))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)

kwargs = {'vmin': vmin, 'vmax': vmax, 'vcenter': 0, 'omit_axes': True, 'omit_cbar': False}

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

vmin = np.min([rgi_diff, rbfi_diff])
vmax = np.max([rgi_diff, rbfi_diff])


print("Plotting differences")

kwargs = {'vmin': vmin, 'vmax': vmax, 'vcenter': None, 'omit_axes': True, 'cmap': 'Reds'}

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

fig.savefig(figs_dir + 'plot.png', dpi=DPI)

print("Done")