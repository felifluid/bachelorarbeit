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

n_r = len(r)
n_theta = len(theta)
r_phi = len(phi)

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

ax.set_xticks(np.linspace(-0.5, 0.5, 5))

q_val = np.max(q[cond])

yticks = [-1-q_val/2, -q_val/2, -1, 0, q_val/2-1, q_val/2]
ax.set_yticks(yticks)
ax.set_yticklabels([r'-q/2-1', r'q/2','-1', '0', r'q/2-1', r'q/2'])

ax.set_xlim(-0.5, 0.5)
ax.set_xlabel("s")
ax.set_ylim(-q_val/2-1, q_val/2)
ax.set_ylabel(r"$\zeta$")

ax.grid(True)

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
b = np.ravel(psi[cond])
c = np.ravel(normed_data)

triangles = topovis.make_regular_triangles(xlen, ylen, periodic=False)

ax.tricontourf(a, b, c, triangles=triangles, cmap=cmap, levels=100)

slc = np.s_[:, :]

psi_ticks = psi[cond][4::7, 0]
ax.set_yticks(psi_ticks)
ax.set_yticklabels(list(map(str, np.round(psi_ticks, 2))))

ax.grid(False, axis='x')
ax.grid(True, axis='y', c=(0,0,0,0.4))

ax.plot(s[cond][slc][:, 1:]*2*np.pi, psi[cond][slc][:, 1:], c=(0,0, 0, 0.6), zorder=5)

s_ticks = s[cond][slc][-1, 1:]
ax.set_xticks(s_ticks * 2*np.pi)
s_ticks_labels = list(map(str, np.round(s_ticks[:-1], 2)))
s_ticks_labels.append(r'$\pm 0.5$')
ax.set_xticklabels(s_ticks_labels)
ax.set_thetalim(-np.pi, np.pi)  # needed to ensure full circle is plotted

save_plot('phi_const/toroidal')

# hamada 3d plot
fig, ax = plt.subplots(1, 1, figsize=figsize, edgecolor='0', subplot_kw={'projection': '3d'})

psi_ticks = np.linspace(np.min(psi[cond]), np.max(psi[cond]), 5)
s_ticks = np.linspace(np.min(s[cond]), np.max(s[cond]), 5)
zeta_ticks = np.linspace(np.min(zeta[cond]), np.max(zeta[cond]), 5)

surf = ax.plot_surface(psi[cond], s[cond], zeta[cond], facecolors=colors, linewidth=1, edgecolor=(0,0,0,0.2), cmap=cmap, rstride=1, cstride=1)

# ax.set_xlim(0, psi_ticks[-1])
# ax.set_ylim(s_ticks[0], s_ticks[-1])
# ax.set_zlim(zeta_ticks[0], zeta_ticks[-1])

ax.set_xticks(psi_ticks)
ax.set_yticks(s_ticks)
ax.set_zticks(zeta_ticks)

ax.set_xticklabels(list(map(str, np.round(psi_ticks, 2))))
ax.set_yticklabels(list(map(str, np.round(s_ticks, 2))))
ax.set_zticklabels(list(map(str, np.round(zeta_ticks, 2))))

ax.set_xlabel(r"$\psi$")
ax.set_ylabel(r"$s$")
ax.set_zlabel(r"$\zeta$")

ax.view_init(30,-40)

save_plot('phi_const/hamada')