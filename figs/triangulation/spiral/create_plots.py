import numpy as np
import matplotlib.pyplot as plt

figs_dir = 'figs/triangulation/spiral/'
ext = '.svg'

c1 = 'lightcoral'

def spiral(phi, a, b=0):
    r = a*phi + b
    x = np.cos(phi) * r
    y = np.sin(phi) * r
    return x, y

N = 200
phi = np.linspace(0, 5*np.pi, N)
np.random.seed(122098129)
#phi = np.random.uniform(0, 6*np.pi, size=N)
a = 1

x1, y1 = spiral(phi, a)

fig, ax = plt.subplots(figsize=(5,5))
ax.axis('off')
ax.set_aspect('equal')
ax.scatter(x1,y1, s=2, c='black', zorder=5)
ax.triplot(x1,y1, lw=0.5, c=c1)

plt.savefig(figs_dir + 'lin' + ext)

noise_factor = 0.1

np.random.seed(122098129)
noisy_x1 = x1 + np.random.normal(0, noise_factor, N)
np.random.seed(121012372)
noisy_y1 = y1 + np.random.normal(0, noise_factor, N)

fig, ax = plt.subplots(figsize=(5,5))
ax.axis('off')
ax.set_aspect('equal')
ax.scatter(noisy_x1,noisy_y1, s=2, c='black', zorder=5)
ax.triplot(noisy_x1,noisy_y1, lw=0.5, c=c1)

plt.savefig(figs_dir + 'noisy' + ext)

phi2 = np.linspace(-np.pi, 3*np.pi, N)
x2, y2 = spiral(phi2, a, np.pi)

x = np.concat((x1, x2))
y = np.concat((y1, y2))

fig, ax = plt.subplots(figsize=(5,5))
ax.axis('off')
ax.set_aspect('equal')
ax.scatter(x1, y1, s=2, c='black', zorder=5)
ax.scatter(x2, y2, s=2, c='black', zorder=5)
ax.triplot(x,y, lw=0.5, c=c1)
plt.savefig(figs_dir + 'interpolation' + ext)

np.random.seed(122095279)
noisy_x2 = x2 + np.random.normal(0, noise_factor, N)
np.random.seed(121012372)
noisy_y2 = y2 + np.random.normal(0, noise_factor, N)

noisy_x = np.concat((noisy_x1, noisy_x2))
noisy_y = np.concat((noisy_y1, noisy_y2))

fig, ax = plt.subplots(figsize=(5,5))
ax.axis('off')
ax.set_aspect('equal')
ax.scatter(noisy_x1,noisy_y1, s=2, c='black', zorder=5)
ax.scatter(noisy_x2, noisy_y2, s=2, c='black', zorder=5)
ax.triplot(noisy_x,noisy_y, lw=0.5, c=c1)
plt.savefig(figs_dir + 'interpolation_noisy' + ext)