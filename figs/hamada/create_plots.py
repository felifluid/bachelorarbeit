import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./scripts')
import topovis

figs_dir = './figs/hamada/'
ext = '.png'
DPI = 200

figsize = (8,6)

plt.rc('font', size=10)

def save_plot(name: str):
    fig = plt.gcf()
    fig.savefig(figs_dir + name + ext, dpi=DPI)

R_ref = 2.5
r = np.linspace(0, 1, 32)
theta = np.linspace(-np.pi, np.pi, 16)
phi = np.linspace(0, 2*np.pi, 64)

r, theta, phi = np.meshgrid(r, theta, phi, indexing='ij')

# construct torus in cartesian geometry

R = R_ref + r*np.cos(theta)
x = R * np.cos(phi)
y = R * np.sin(phi)
z = r * np.sin(theta)

s_B = 1
s_J = 1

# calculate hamada coordinates

psi = r/R_ref
s = 1/(2*np.pi)*(theta + psi*np.sin(theta))
q = 15*psi**2 + 1
zeta = (- phi/(2*np.pi) + s_B*s_J*np.abs(q)/np.pi * np.arctan(np.sqrt((1-psi)/(1+psi)) * np.tan(theta/2)))

# slice only the biggest psi
cond = np.s_[-1,:,:]

xlen, ylen = np.shape(x[cond])  # this is not neccessary "x" or "y" related, just dummy values

cmap = plt.get_cmap('plasma')   # set cmap


data = zeta[cond]   # which data to visualize as colors?

normed_data = (data-np.min(data))/(np.max(data)-np.min(data))

# calculate colors

colors = np.empty(shape=(xlen, ylen, 4))
for i in range(xlen):
    for j in range(ylen):
        val = float(normed_data[i,j])
        color = cmap(val)
        colors[i, j] = color

# make toroidal 3d plot

fig, ax = plt.subplots(1, 1, figsize=figsize, edgecolor='0', subplot_kw={'projection': '3d'}, layout='compressed')
plt.tight_layout()

surf = ax.plot_surface(x[cond], y[cond], z[cond], facecolors=colors, linewidth=1, edgecolor=(0,0,0,0.2), cmap=cmap, rstride=1, cstride=1)

ax.set_aspect('equal')

# adjust viewing angle
# ax.view_init()

# disable axis
# ax.set_axis_off()
ax.set_zticks([-1, -0.5, 0, 0.5, 1])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.tight_layout()
save_plot('psi_const/toroidal')

# make hamada 2d plot

fig, ax = plt.subplots(1, 1, figsize=figsize)

# draw black border around contour
ax.plot(s[cond][:,0], zeta[cond][:,0], c='0')
ax.plot(s[cond][:,-1], zeta[cond][:,-1], c='0')
ax.plot(s[cond][0,:], zeta[cond][0,:], c='0')
ax.plot(s[cond][-1,:], zeta[cond][-1,:], c='0')

a = np.ravel(s[cond])
b = np.ravel(zeta[cond])
c = np.ravel(normed_data)

triangles = topovis.make_regular_triangles(xlen, ylen)

surf = ax.tricontourf(np.ravel(s[cond]), np.ravel(zeta[cond]), np.ravel(normed_data), triangles=triangles, cmap=cmap, levels=100, zorder=-5)

# plot grid
# ax.plot(s[-1, :, ::10], zeta[-1, :, ::10], c=(0,0,0,0.2))

ax.set_xlim(-0.6, 0.6)
ax.set_xlabel("s")
ax.set_ylim(-2.8, 1.8)
ax.set_ylabel(r"$\zeta$")
save_plot('psi_const/hamada')


#### CONST PHI PLOT

cond = np.s_[:,:,0]

xlen, ylen = np.shape(x[cond])  # this is not neccessary "x" or "y" related, just dummy values

cmap = plt.get_cmap('plasma')   # set cmap

data = zeta[cond]   # which data to visualize as colors?

normed_data = (data-np.min(data))/(np.max(data)-np.min(data))

# calculate colors

colors = np.empty(shape=(xlen, ylen, 4))
for i in range(xlen):
    for j in range(ylen):
        val = float(normed_data[i,j])
        color = cmap(val)
        colors[i, j] = color


fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize)

a = np.ravel(theta[cond])
b = np.ravel(r[cond])
c = np.ravel(normed_data)

triangles = topovis.make_regular_triangles(xlen, ylen, periodic=False)

ax.tricontourf(a, b, c, triangles=triangles, cmap=cmap, levels=100)


ax.set_yticks([])

ax.grid(False)

slc = np.s_[:, :]

ax.plot(s[cond][slc][:, 1:]*2*np.pi, r[cond][slc][:, 1:], c=(0,0, 0, 0.4), zorder=5)

s_ticks = s[cond][slc][-1, 1:]
ax.set_xticks(s_ticks * 2*np.pi)
s_ticks_labels = list(map(str, np.round(s_ticks[:-1], 2)))
s_ticks_labels.append(r'$\pm 0.5$')
ax.set_xticklabels(s_ticks_labels)
ax.set_thetalim(-np.pi, np.pi)  # needed to ensure full circle is plotted

save_plot('phi_const/toroidal')

# hamada 3d plot
fig, ax = plt.subplots(1, 1, figsize=figsize, edgecolor='0', subplot_kw={'projection': '3d'})

surf = ax.plot_surface(psi[cond], s[cond], zeta[cond], facecolors=colors, linewidth=1, edgecolor=(0,0,0,0.2), cmap=cmap)

ax.set_xlabel(r"$\psi$")
ax.set_ylabel(r"$s$")
ax.set_zlabel(r"$\zeta$")

save_plot('phi_const/hamada')