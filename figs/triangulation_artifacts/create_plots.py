import sys
sys.path.append('./scripts')
import topovis
import h5py
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def plot_subsection(r, z, pot, xlim, ylim, grid: bool, method, **kwargs):
    r_flat = np.ravel(r)
    z_flat = np.ravel(z)
    pot_flat = np.ravel(pot)

    nx, ns = np.shape(r)

    fig, ax = plt.subplots()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xticks([])
    ax.set_yticks([])

    if method == 'regular':
        triangles = topovis.make_regular_triangles(nx, ns, periodic=True)
    elif method == 'delaunay':
        triangles = tri.Triangulation(r_flat, z_flat).triangles
    
    if grid:
        ax.triplot(r_flat, z_flat, triangles=triangles, lw=0.25, c='0.3')
        ax.scatter(r_flat, z_flat, c='r', marker='.', edgecolors='0', lw=0, s=50)
    else:
        kwargs['levels'] = 200
        kwargs['cmap'] = 'seismic'

        ax.tricontourf(r_flat, z_flat, pot_flat, triangles=triangles, **kwargs)
        ax.plot(np.ravel(r[-1, :]), np.ravel(z[-1, :]), color='grey', alpha=0.3, linewidth=1, zorder=11)
        ax.plot(np.ravel(r[0, :]), np.ravel(z[0, :]), color='grey', alpha=0.3, linewidth=1, zorder=11)
        ax.fill(r[0, :], z[0, :], color='white', zorder=10)

    return fig, ax

figs_dir = './figs/triangulation_artifacts/'
data_dir = './data/lin/chease/bugfix/ns128/'

def figpath(filename, ext):
    return figs_dir+filename+'.'+ext

path = pathlib.Path(data_dir+'topovisdata.h5')

if not path.is_file():
    topovis.main(['-vv', '--triang-method', 'regular', '--data-out', path.as_posix(), '--omit-axes', data_dir+'gkwdata.h5', '--fx', '1', '--fs', '1'])

with h5py.File(path.as_posix()) as f:
    R_FLAT = f['r_n'][()]
    Z_FLAT = f['z'][()]
    POT_FLAT = f['pot'][()]

    X = f['x'][()]
    S = f['s'][()]

NX = len(X)
NS = len(S)

R = np.reshape(R_FLAT, (NX,NS))
Z = np.reshape(Z_FLAT, (NX,NS))
POT = np.reshape(POT_FLAT, (NX,NS))

slc = np.s_[0:None:4, :]
xlim = (1.245, 1.26)
ylim = (-0.02, 0.092)

kwargs = {}
fig, ax = plot_subsection(R[slc], Z[slc], POT[slc], xlim=xlim, ylim=ylim, grid=True, method='delaunay', **kwargs)
fig.savefig(figpath('sparse/delaunay/grid','png'))

fig, ax = plot_subsection(R[slc], Z[slc], POT[slc], xlim=xlim, ylim=ylim, grid=False, method='delaunay', **kwargs)
fig.savefig(figpath('sparse/delaunay/contour','png'), dpi=300)

fig, ax = plot_subsection(R[slc], Z[slc], POT[slc], xlim=xlim, ylim=ylim, grid=True, method='regular', **kwargs)
fig.savefig(figpath('sparse/regular/grid','png'))

fig, ax = plot_subsection(R[slc], Z[slc], POT[slc], xlim=xlim, ylim=ylim, grid=False, method='regular', **kwargs)
fig.savefig(figpath('sparse/regular/contour','png'), dpi=300)

# sheared

xlim = (0.65, 0.67)
ylim = (0.015, 0.045)
slc = np.s_[:, 100:130]

fig, ax = plot_subsection(R[slc], Z[slc], POT[slc], xlim=xlim, ylim=ylim, grid=True, method='delaunay')
fig.savefig(figpath('sheared/delaunay/grid','png'))

fig, ax = plot_subsection(R[slc], Z[slc], POT[slc], xlim=xlim, ylim=ylim, grid=False, method='delaunay')
fig.savefig(figpath('sheared/delaunay/contour', 'png'), dpi=300)

fig, ax = plot_subsection(R[slc], Z[slc], POT[slc], xlim=xlim, ylim=ylim, grid=True, method='regular')
fig.savefig(figpath('sheared/regular/grid', 'png'))

fig, ax = plot_subsection(R[slc], Z[slc], POT[slc], xlim=xlim, ylim=ylim, grid=False, method='regular')
fig.savefig(figpath('sheared/regular/contour', 'png'), dpi=300)