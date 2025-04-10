import sys
sys.path.append('scripts')
import topovis
import matplotlib.pyplot as plt
import numpy as np

figs_dir = 'figs/boundary_conditions/zeta_s/'
ext = '.svg'
dat = topovis.GKWData('data/circ/ns128/gkwdata.h5')

zeta_s = topovis.shift_zeta(g=dat.g, s=dat.s, phi=0, geom_type='circ')

fig, ax = plt.subplots(1,1, figsize=(5,5))

ax.set_xlabel("s")
ax.set_ylabel(r"$\zeta$")

n = 64

s_e = topovis.extend_regular_array(dat.s, n)
zeta_s_p = topovis.extend_periodically(zeta_s, n, 1)
q = dat.q[-1]

out_args = {'c': 'C1', 'alpha': 0.5}
in_args = {'c': 'C1'}

slc = np.s_[:n]
zeta_s_p[-1, slc] -= dat.q[-1, None]
ax.plot(s_e[slc], zeta_s_p[-1, slc], **out_args)

slc = np.s_[n:-n]
ax.plot(s_e[slc], zeta_s_p[-1, slc], label=r"$\zeta$-shift",**in_args)

slc = np.s_[-n:]
zeta_s_p[-1, slc] += dat.q[-1, None]
ax.plot(s_e[slc], zeta_s_p[-1, slc], **out_args)

ax.set_xlim((-1, 1))
ax.set_ylim((-q, q))
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([-q, -q/2, 0, q/2, q])
ax.set_yticklabels(["-q", "-q/2", "0", "q/2", "q"])

ax.axhline(-q/2, color='0', linestyle='dashed')
ax.axhline(q/2, color='0', linestyle='dashed')
ax.axvline(-0.5, color='0', linestyle='dashed')
ax.axvline(0.5, color='0', linestyle='dashed')
ax.grid()

ax.legend()

plt.savefig(figs_dir + 'q-scale' + ext)

fig, ax = plt.subplots(figsize=(5,5))

ax.set_xlabel("s")
ax.set_ylabel(r"$\zeta$")

ax.set_xlim((-1, 1))
ax.set_ylim((-0.1, 1.1))
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

ax.axhline(1, color='0', linestyle='dashed')
ax.axhline(0, color='0', linestyle='dashed')
ax.axvline(-0.5, color='0', linestyle='dashed')
ax.axvline(0.5, color='0', linestyle='dashed')
ax.grid()

zeta_s_p = zeta_s_p % 1

diffs = np.diff(zeta_s_p[-1, :])
jumps = np.ravel((diffs < 0).nonzero()) + 1

zeta_s_p_cut = np.insert(zeta_s_p[-1,:], jumps, np.nan)
s_e_cut = np.insert(s_e, jumps, np.nan)


ax.plot(s_e_cut[n:-n], zeta_s_p_cut[n:-n], c='C1', label=r"$\zeta$-shift")
ax.plot(s_e_cut[-n:], zeta_s_p_cut[-n:], c='C1', alpha=0.5)
ax.plot(s_e_cut[:n], zeta_s_p_cut[:n], c='C1', alpha=0.5)

ax.legend()

plt.savefig(figs_dir + 'zeta-scale' + ext)


