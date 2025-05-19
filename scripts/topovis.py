################################################### IMPORTS ##################################################

import logging
import argparse
import sys
import matplotlib.cm
import numpy as np
import h5py
import enum
import scipy.integrate
import scipy.interpolate
import matplotlib.tri
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors

################################################## FUNCTIONS #################################################


def transform_indice(xi, si, ns):
    return si + xi * ns


def array_slice(a, axis, slc : slice):
    """
    Neat little helper function to create a sliced view of an ndarray on a dynamic axis.
    This avoids creating a copy caused by nparray.take().
    Published by Leland Hempworth under CC BY-SA 4.0 Licence (https://stackoverflow.com/a/64436208), slightly modified.
    """
    return a[(slice(None),) * (axis % a.ndim) + (slc,)]


def check_regular_spacing(a, tol=1e-9) -> tuple[bool, float]:
    diffs = np.diff(a)
    if np.allclose(diffs, diffs[0], atol=tol):
        isRegular = True
    else:
        isRegular = False
    
    return isRegular, float(np.mean(diffs, dtype=float))

def extend_periodically(a, overlap : int, axis : int | list[int]) -> np.ndarray:
    """
    Wraps ndarray periodically along given singular axis or mulitple axes.

    Params
    ------
    a : ndarray
        ndarray to wrap around
    overlap : int
        how many values to wrap around
    axis : int | list[int]
        along which axis to wrap array around.

    Returns
    -------
    Array with wrapped around values.

    Examples
    --------
    ```
    arr = [0,1,2,3,4,5]
    periodic_wrap_values(arr, overlap=2, axis=0)
    » [4,5,0,1,2,3,4,5,0,1]
    ```
    """
    axs = [axis] if isinstance(axis, int) else axis
    for ax in axs:
        before = array_slice(a, ax, slice(-overlap, None))
        after = array_slice(a, ax, slice(overlap))
        a = np.concatenate((before, a, after), axis=ax)
    return a
    
def extend_regular_array(a : np.typing.ArrayLike, n : int):
    """
    Extends a regularly (equally) spaced array in both directions.  
    The new array will have `n` values prepended and apended equally distant.

    Parameters
    ----------
    a : ArrayLike
    n : int
        How many points to prepend and append.
    """
    a = np.asarray(a)
    is_regular, step = check_regular_spacing(a)
    if not is_regular:
        raise ValueError('The array is not equally spaced')
    step = a[1] - a[0]
    return np.linspace(a[0]-step*n, a[-1]+step*n, len(a)+(2*n))    

def extend_regular_grid(grid : list, n : int, axis : int | list[int]):
    """
    This build onto `extend_regular_array` to work for ndarrays on multiple axes.
    """
    axs = [axis] if isinstance(axis, int) else axis
    for ax in axs:
        grid[ax] = extend_regular_array(grid[ax], n)
    return grid


def interpolate_regular_array(a : np.typing.ArrayLike, f : int):
    """
    Parameters
    ----------
    a : ArrayLike
        array to interpolate
    f : int
        scaling factor
    extrapolate : bool
        whether to extrapolate outside of domain

    Returns
    -------
    out : ArrayLike, 1d
        original array with `f-1` additional gridpoints interpolated inbetween, shape (n+(f-1)*(n-1),)
    """

    if not isinstance(f, int):
        raise ValueError(f"fx must be an integer.")
    
    isRegular, spacing = check_regular_spacing(a)
    if not isRegular:
        raise ValueError(f"array is not equally spaced")

    if f < 1:
        raise ValueError(f"fx must be at least 1, but found {f}")
    elif f == 1:
        # factor is one; just return original array
        return a
    else:
        n = len(a)
        n_fine = n + (f-1) * (n-1)
        return np.linspace(a[0], a[-1], n_fine)

def extrapolate_s_grid(s : np.typing.ArrayLike, fs : int, include_boundaries='hi'):
    """
    Interpolate and extrapolate the s-grid. 

    This makes a few assumptions about the s-grid.

        - The grid is defined between `-0.5+(spacing/2)` and `+0.5-(spacing/2)`.  \
        This means there is excactly `spacing` between the first and last grid-point assuming periodicity.

    The fine grid will still be within the boundaries of `[-0.5, 0.5]`

    Parameters
    ----------
    s : ArrayLike
        the sparse s-grid
    fs : int
        scaling factor
    include_boundaries : 'hi', 'lo', 'both', 'none'
        only gets applied with even `fs` to choose which boundary value to use.  \
        With even `fs` 'both' will extrapolate both `-0.5` AND `0.5`,  \
        which is sometimes not wanted as the extrapolated as grid will have a length of `(ns*fs)+1` \
        instead of `ns*fs`.
        Both 'hi' and 'lo' work around this, by just including either `0.5` OR `-0.5`.
        There is also the option of include neither of them.

    Returns
    -------
    s_fine : ArrayLike
        The fine extrapolated s-grid
    offsets : tuple[int, int]
        Additionally export both `offset_lo` and `offset_hi` as markers which values are outside the sparse grid.  
        `s_fine[:offset_lo]` and `s_fine[-offset_hi:None]` will be the extrapolated values respectively.
    """

    if not isinstance(fs, int):
        raise ValueError(f"fx must be an integer.")
    
    isRegular, spacing = check_regular_spacing(s)
    if not isRegular:
        raise ValueError(f"array is not equally spaced")

    if fs < 1:
        raise ValueError(f"fx must be at least 1, but found {fs}")
    elif fs == 1:
        # factor is one; just return original array
        offset_lo = 0
        offset_hi = 0
        s_fine = np.asarray(s, dtype=float)
    else:
        ns = len(s)
        ns_fine = ns * fs
        spacing_fine = spacing / fs
        if fs % 2 == 0: # fs even
            # both -0.5 AND 0.5 describe the same point on a poloidal slice
            offset_lo = (fs//2)
            offset_hi = (fs//2)

            if include_boundaries == 'lo':
                offset_hi -= 1
            elif include_boundaries == 'hi':
                offset_lo -= 1
            elif include_boundaries == 'none':
                offset_hi -= 1
                offset_lo -= 1

            s_lo = s[0] - spacing_fine * offset_lo
            s_hi = s[-1] + spacing_fine * offset_hi
            s_fine = np.linspace(s_lo, s_hi,  ns_fine, dtype=float)
        else: # fs odd
            # -0.5 and 0.5 are not included
            # fine_grid is symmetrical
            offset_lo = (fs-1)//2
            offset_hi = (fs-1)//2
            s_lo = s[0] - spacing_fine * offset_lo
            s_hi = s[-1] + spacing_fine * offset_hi
            s_fine = np.linspace(s_lo, s_hi,  ns_fine, dtype=float)
        
    return s_fine

def interpolate_hamada_grid(x : np.typing.ArrayLike, s : np.typing.ArrayLike, fx : int, fs : int, extrapolate_s=False, include_boundaries='hi'):
    """
    Interpolates the Hamada (psi, s)-grid.

    Args:
        x (ArrayLike): The sparse psi-grid.
        s (ArrayLike): The sparse s-grid.
        fx (int): Scaling factor for psi.
        fs (int): Scaling factor for s.
        extrapolate_s (bool): Whether to extrapolate the s-grid or not. This has no effect on the psi-grid.

    Returns:
        x_fine (ArrayLike): The fine psi-grid.
        s_fine (ArrayLike): The fine s-grid.
    """

    x_fine = interpolate_regular_array(x, fx)
    
    if extrapolate_s:
        s_fine = extrapolate_s_grid(s, fs, include_boundaries)
    else:
        s_fine = interpolate_regular_array(s, fs)

    return x_fine, s_fine

def grid_to_points(grid):
    """
    Parameters
    ----------
    X1, X2,..., XN : tuple of N ndarrays

    Returns
    -------
    out : ndarray, shape (-1, N)

    Reference
    ---------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html#scipy.stats.gaussian_kde
    """
    return np.vstack(list(map(np.ravel, grid))).T
    
def shift_zeta(g, s, phi, geom_type, q=None, r_n=None, r_ref=None, nx=None, ns=None):
    """
    Generate the grid for zeta which preserves phi = const. 

    Parameters
    ----------
    g : ArrayLike
        gmap, shape (nx, ns)
    s : ArrayLike
        The grid of s values
    phi : float
        Phi value to preserve for the zeta-shift
    geomtype : str
        The type of geometry. Can be 'circ', 's-alpha', or 'chease_global'.
    sign_b : float
        The sign value of B. Only needed for s-alpha geometry.
    sign_j : float 
        The sign value of J. Only needed for s-alpha geometry.
    q : ArrayLike
        The safety-factor q. Only needed for s-alpha geometry.
    r_n : ndarray
        The normalized major radius R_N, shape (nx, ns). Only needed for chease-global geometry.
    r_ref : float
        Reference value of R. Only needed for chease-global geometry.
    nx : int
        Number of values in psi direction. Only needed for chease-global geometry.
    ns : int
        Number of values in s direction. Only needed for chease-global geometry.
        
    Returns
    -------
    out : ndarray
        The zeta-shift grid which is phi-preserving. It will be mapped back to `0 <= zeta <= 1`.

    Notes
    -----
    - For 'circ' geometry, the zeta shift is calculated as (-phi_val / (2 * pi) + g) % 1
    - For 's-alpha' geometry, the zeta shift is calculated as (signB*signJ/(2*pi)*(2*pi*abs(q_val)*s-phi_val)) % 1.
    - For 'chease_global' geometry, the zeta shift is calculated using numerical integration (trapezoidal rule).  \
    zeta(s) = -phi/2pi + gmap * integral_0^s (1/(R**2)). Also: R is not the unitless R as for circ geom, but R_N*R_REF.
    """
    # TODO: extract shift-factor 'G' out of this function.
    # circ:
    #   G = gmap - qs
    # chease-global: 
    #   G = gmap * chease_factor - qs
    # s-alpha:
    #   G = 0
    if geom_type == 'circ' or geom_type == 'universal':
        zeta = -phi / (2*np.pi) + g
    elif geom_type == 's-alpha':
        #FIXME: equation for s-alpha is not right
        zeta = (phi) / (2*np.pi) * q * s
    elif geom_type == 'chease_global':
        zeta = -phi/(2*np.pi) + g * chease_integration(s, r_n, r_ref, nx, ns)
    else:
        raise ValueError(f"geom_types other than 'circ', 's-alpha' and 'chease_global' are not supported")
    return zeta

def chease_integration(s, r_n, r_ref, nx, ns):
    xi = np.arange(nx)
    si = np.arange(ns)

    r = r_n * r_ref

    # number of negative elements in s
    n_neg = ns//2
    # s0 is placed after n_neg into array
    s0_idx = n_neg + 1
    
    if ns % 2 == 0 :
        # s=0 doesn't exist > therefore R(s=0) also doesn't exist
        # interpolate R(s=0)
        # TODO: it would probably suffice to just take the nearest n points surrounding s=0
        r_interp = scipy.interpolate.RegularGridInterpolator((xi, si), r, 'quintic')

        # evaluate R(s=0) for all psi
        r0 = r_interp((xi,0))
        r = np.insert(r, n_neg, r0, axis=1) # shape (NX, NS+1)

        s0 = np.array([0])
        s = np.insert(s, n_neg, s0)

    res = np.zeros((nx, len(s)))   # S is either Ns or Ns+1

    # TODO: maybe this could be done in 2D/3D instead of for-loops
    # use of cumulative_trapezoid func could also be interesting
    # maybe there's a more elegant solution which doesn't need the array split in pos and neg
    for j in range(nx):
        for i in range(s0_idx): # S < 0
            y = 1 / r[j, i:s0_idx] ** 2
            x = s[i:s0_idx]
            res[j][i] = -scipy.integrate.trapezoid(y, x)
        
        for i in range(s0_idx, len(s)): # S > 0
            y = 1 / r[j, s0_idx:i] ** 2
            x = s[s0_idx:i]
            res[j][i] = scipy.integrate.trapezoid(y, x)
    # remove s0 element
    return np.delete(res, s0_idx, 1)

def calculate_potential(fcoeffs, zeta, k) -> np.ndarray:
    """
    Parameters
    ----------
    fcoeffs : ndarray
        The complex fourier coefficients.
    zeta : ndarray
        The zeta grid.
    k : int
        The wave vector.
    
    Returns
    -------
    pot : ndarray
        The real potential.
    """
    # exponential terms
    exp_pos = np.exp(1j * k * zeta)
    exp_neg = np.exp(-1j * k * zeta)

    return (fcoeffs * exp_pos + np.conjugate(fcoeffs) * exp_neg).real

def make_zeta_grid(mphi : int, n_spacing : int):
    """
    Make zeta grid for whole torus (not just 1/n_spacing of torus). This will only be used in the nonlinear case. 

    Args
    ----
    mphi : int
        Number of grid points in Zeta direction for simulated part of torus. 
    n_spacing : int
        Smallest mode number of toroidal modes that are resolved in the simulation.
    
    Returns
    -------
    zeta_grid: ArrayLike, shape (mphi*n_spacing)
        The zeta grid.
    """
    L_zeta = 1/n_spacing
    delta_zeta = L_zeta / mphi
    zeta = np.arange(mphi*n_spacing)*delta_zeta
    return zeta

def repeat_pot(pot, n_spacing : int):
    """
    Returns the potential on the whole torus as potential_data is only for (1/nspacing)-part of the torus.

    Args
    ----
    pot : ndarray
        The 3D potential data for in GKW simulated part of torus.
    n_spacing : int
        The number of torus sections.
    
    Returns
    -------
    whole_pot : ndarray
        The potential on the whole torus.
    """

    return np.tile(pot, (1,1,n_spacing))
# ---------------------------------------------- PLOTTING ------------------------------------------------

def make_regular_triangles(nx, ns, dir='r', periodic=False):
    """
    Args
    ----
    nx : int
        number of grid points in psi-direction
    ns : int
        number of grid points in s-direction
    dir : str
        direction for drawing the diagonal line. either 'l' or 'r'
    periodic : bool
        whether triangles should be added between the first and last s-grid points.
    
    Returns
    -------
    triangles : ndarray, shape (2*(n[-1]*m-1), 3), n gets subtracted by one if `periodic` is `False`.
        Array representing a regular triangle grid with counter-clockwise indices. The values are referencing indices of a flattened x-s-grid.
    
    Examples
    --------
    >>> make_regular_triangle_grid(nx=2, ns=3, dir='r', periodic=False) 
    >>> 0 - 1 - 2
    >>> | / | / |
    >>> 3 - 4 - 5

    >>> make_regular_triangle_grid(nx=2, ns=3, dir='l', periodic=True) 
    >>> 0  - 1  - 2  - 0
    >>> | \\ | \\ | \\ |
    >>> 3  - 4  - 5  - 3
    """

    # check args
    if nx < 2 : raise ValueError(f'nx has to be at least 2, got {nx}')
    if ns < 2 : raise ValueError(f'ns has to be at least 2, got {nx}')
    if dir != 'l' and dir != 'r': raise ValueError(f"dir must be either 'l' or 'r', got {dir}")

    # construct indices grid
    xi = np.arange(nx)
    si = np.arange(ns)

    ss, xx = np.meshgrid(si, xi)    # has to be in this order
    # ss = [[0,1,2,..., ns-1], [0,1,2, ..., ns-1], ... [   0,    1,    2, ... ns-1]]
    # xx = [[0,0,0,...,    0], [1,1,1, ...,    1], ... [nx-1, nx-1, nx-1, ... nx-1]]

    if periodic:
        # concat first s points to end 

        ss = np.concatenate((ss, np.zeros(shape=(nx,1), dtype=int)), axis=1)
        # ss = [[0,1,2,..., ns-1, 0], [0,1,2, ..., ns-1, 0], ... [   0,    1,    2, ... ns-1,    0]]

        xx = np.concatenate ((xx, np.reshape(xi, (nx, 1))), axis=1)
        # xx = [[0,0,0,...,    0, 0], [1,1,1, ...,    1, 1], ... [nx-1, nx-1, nx-1, ... nx-1, nx-1]]

    # a - b
    # | X |
    # c - d

    # NOTE: contrary to the name 'flat' this is still a 2D-array
    # this translates every pair of xi and si indices to one index j in a *flattened* x or s array.
    flat = ss + xx * ns

    # this doesn't copy values and just creates views of the original array
    a = flat[   :-1,  :-1]
    b = flat[   :-1, 1:  ]
    c = flat[1  :  ,  :-1]
    d = flat[1  :  , 1:  ]

    # flatten arrays in preparation for np.column_stack
    a_flat = np.ravel(a)
    b_flat = np.ravel(b)
    c_flat = np.ravel(c)
    d_flat = np.ravel(d)
    
    if dir == 'r':
        triangles = np.concat((np.column_stack((a_flat, c_flat, b_flat)), np.column_stack((b_flat, c_flat, d_flat))))
    elif dir == 'l':
        triangles = np.concat((np.column_stack((a_flat, c_flat, d_flat)), np.column_stack((a_flat, d_flat, b_flat))))

    return triangles

def _plot_contourf(ax, triangulation, z, show_grid=False, **kwargs):
    x = triangulation.x
    y = triangulation.y
    cmap = kwargs.pop('cmap', 'seismic')
    levels = kwargs.pop('levels', 200)

    ax.tricontourf(triangulation, z, levels=levels, zorder=0, cmap=cmap, **kwargs)
    if show_grid : _plot_grid(ax, triangulation, z, **kwargs)
    return ax

def _plot_grid(ax, triangulation, z, **kwargs):
    ax.triplot(triangulation, c=(0,0,0,0.2), lw=0.01)
    ax.scatter(triangulation.x, triangulation.y, 
               marker='.', 
               c=z, 
               s=0.1, 
               edgecolors='none', 
               zorder=2, 
               cmap=kwargs.pop('cmap', 'seismic'),
               vmin=kwargs.pop('vmin'),
               vmax=kwargs.pop('vmax')
            )
    
def _clip_cmap(cmap, vmin, vmax, vcenter=None, vmin_clip=None, vmax_clip=None):
    """
    Return a clipped but color-mapping preserving Norm and Colormap.

    The returned Norm and Colormap map data values to the same colors as
    would  `Normalize(vmin, vmax)`  with *cmap_name*, but values below
    *vmin_clip* and above *vmax_clip* are mapped to under and over values
    instead.

    Ref: https://discourse.matplotlib.org/t/limiting-colormapping/20598/5
    """
    if vmin_clip is None:
        vmin_clip = vmin
    if vmax_clip is None:
        vmax_clip = vmax
    
    assert vmin <= vmin_clip < vmax_clip <= vmax
    cmin = (vmin_clip - vmin) / (vmax - vmin)
    cmax = (vmax_clip - vmin) / (vmax - vmin)

    big_cmap = matplotlib.cm.get_cmap(cmap, 512)

    new_norm = matplotlib.colors.Normalize(vmin_clip, vmax_clip)
    new_cmap = matplotlib.colors.ListedColormap(big_cmap(np.linspace(cmin, cmax, 256)))

    return new_cmap, new_norm

def plot(r, z, pot, fig=None, ax=None, triang_method='regular', omit_axes=False, omit_cbar=False, plot_grid=False, **kwargs):
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    vmin = kwargs.pop('vmin', np.min(pot))
    vmax = kwargs.pop('vmax', np.max(pot))
    vcenter = kwargs.pop('vcenter', 0.0)
    cmap = kwargs.pop('cmap', 'seismic')
    
    nx, ns = np.shape(r)

    r_flat = np.ravel(r)
    z_flat = np.ravel(z)

    if triang_method == 'regular':
        triangles = make_regular_triangles(nx, ns, periodic=True)
    elif triang_method == 'delaunay':
        triangles = matplotlib.tri.Triangulation(r_flat, z_flat).triangles
    else:
        raise ValueError(f"No other triangulation method supported other than 'regular' and 'delaunay', got {triang_method}")

    if omit_axes:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel(r'$R$')
        ax.set_ylabel(r'$Z$')


    ax.set_aspect(1, adjustable='datalim')
    
    triangulation = matplotlib.tri.Triangulation(r_flat, z_flat, triangles=triangles)

    # plot closed line at x=0 and x=-1 (inner and outer radial border of data)
    ax.plot(np.append(r[0, :], r[0,0]), np.append(z[0, :], z[0,0]), ls='-', c=(0,0,0, 0.8), lw=0.2, zorder=11)
    ax.plot(np.append(r[-1, :], r[-1,0]), np.append(z[-1, :], z[-1,0]), ls='-', c=(0,0,0, 0.8), lw=0.2, zorder=11)

    # fill inner empty circle with white area, this is only neccessary in case of delaunay triangulation
    ax.fill(r[0, :], z[0, :], color='white', zorder=10)
    
    if plot_grid:
        _plot_grid(ax, triangulation, pot)
    else:
        _plot_contourf(ax, triangulation, pot, show_grid=False, vmin=vmin, vmax=vmax, cmap=cmap)

    if not omit_cbar:
        vmin_clip = np.min(pot)
        vmax_clip = np.max(pot)
        
        clipped_cmap, clipped_norm = _clip_cmap(cmap, vmin, vmax, vcenter, vmin_clip, vmax_clip)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=clipped_norm, cmap=clipped_cmap), ax=ax)

    return fig, ax

def parse_args(args):
    parser = argparse.ArgumentParser(description='Interpolation of a poloidal cross section for given toroidal angle phi and time step for potential data. Zonal or non-zonal potential is displayed.')

    parser.add_argument('-v', '--verbose', 
                        action='count', 
                        default=0,
                        help='Causes script to print debugging messages about its progress. Multiple -v increase the verbosity to a maximum of 3.')
    parser.add_argument('hdf5_filepath', 
                        type=str,
                        help='File path to hdf5 File.')
    parser.add_argument('--phi', 
                        type=float,
                        default=0.0, 
                        help='(optional) Value for toroidal angle phi. Default is 0')
    parser.add_argument('--poten-timestep', 
                        type=int, 
                        nargs=1,
                        help='(optional) Time step for potential. If not specified the last timestep is being used.')
    parser.add_argument('-z', '--zonal',
                        action='store_true',
                        dest='zonal',
                        help="(optional) If set plots zonal potential. This is only being used in non-linear simulations.")
    parser.add_argument('--legacy-gmap',
                        action='store_true',
                        dest='legacy_gmap',
                        help="Calculates the G-factor numerically, instead of just importing from GKW." #TODO: Better description
                        )
    parser.add_argument('--dsf', '--downsample',
                        type=int,
                        default=0,
                        help="(optional) Downsamples the s-grid resolution by just using every Nth grid point. Used for benchmarking and debugging interpolation and triangulation. This is an experimental feature."
                        )
    plot_group = parser.add_argument_group('plot', description='Plotting parameters.')
    plot_group.add_argument('--triang-method', 
                            type=str,
                            choices=('regular', 'delaunay'),
                            dest='triang_method',
                            default='regular', 
                            help="(optional) Method to use for triangulation. 'regular' creates regular triangles in hamada coordinates, whereas 'delaunay' performs a Delaunay triangulation in poloidal coordinates. Default is 'regular'.")
    plot_group.add_argument('-p', '--plot-out',
                            dest='plot_out',
                            type=str,
                            help="(optional) Specify a file to write the plot into WITH extension. Can either be a full path, or a filename. If it's a filename, it will save to the current directory. Default is None."
                            )
    plot_group.add_argument('--plot-grid',
                            dest='plot_grid',
                            action='store_true',
                            help='(optional) plots scatter plot with triangulation instead of a contour plot. Good for debugging, spotting triangulation artifacts and getting a feel for grid density.'
                            )
    plot_group.add_argument('--levels',
                            dest='levels',
                            nargs=1,
                            type=int,
                            default=200,
                            help="(optional) How many levels to use for the tricontourf plot. Default is 200. Gets omitted when combined with --plot-grid"
                            )
    plot_group.add_argument('--dpi',
                            dest='dpi',
                            type=int,
                            default=400,
                            help="(optional) How many DPI should the resulting png have? Has no effect when plotfile type is 'pdf' or 'svg'. Default is 400."
                            )
    plot_group.add_argument('--omit-axes',
                            dest='omit_axes',
                            action='store_true',
                            help="(optional) Hide axes in plot."
                            )
    interpolation_group = parser.add_argument_group('interpolation', description='Interpolation parameters.')
    interpolation_group.add_argument('--fx', 
                                     type=int,
                                     default=1,
                                     help='(optional) Factor by which to refine the psi-grid through interpolation.')
    interpolation_group.add_argument('--fs', 
                                     type=int, 
                                     default=1,
                                     help='(optional) Factor by which to refine the s-grid through interpolation.')
    interpolation_group.add_argument('--interpolator',
                                     dest='interpolator',
                                     choices=('rgi', 'rbfi'),
                                     default='rgi',
                                     help="(optional) Which interpolator to use to interpolate the potential. 'rgi' uses the RegularGridInterpolator, which interpolates on hamada coordinates, 'rbfi' uses the RBFInterpolator, which interpolates in poloidal coordinates. Default is 'rgi'.")
    interpolation_group.add_argument('-m', '--method',
                                     metavar='method',
                                     choices=('nearest', 'linear', 'cubic', 'quintic'),
                                     default='cubic',
                                     help="(optional) Method of interpolation. Valid options are: 'nearest', 'linear', 'cubic', 'quintic'. Default is 'cubic'. When faced with memory or processing constraints this should be changed to 'linear'.")
    interpolation_group.add_argument('-d', '--data-out',
                                     dest='data_out',
                                     type=str, 
                                     help="(optional) Specify a h5-file to write the interpolation data into. Can either be a full path, or a filename. If it's a filename, it will save to the current directory.")

    return parser.parse_args(args)
################################################# CLASSES ################################################

class GKWData:
    """
    Class which handles importing of `gkwdata.h5` files.
    """
    
    class ParallelDatQuantities(enum.Enum):
        """
        The column order of the `parallel.dat` quantities as specified in module `diagnos_mode_struct`.

        Reference
        ---------
        GKW Documentation - Chapter 10.4 (p. 144)
        """

        def col(self):
            """
            Adjust to index 0.

            Returns
            -------
            out : int
                `self.value - 1`.
            """
            return self.value - 1

        SGR = 1
        PHI_REAL = 2
        PHI_IM = 3
        APAR_REAL = 4
        APAR_IM = 5
        DENS_REAL = 6
        DENS_IM = 7
        EPAR_REAL = 8
        EPAR_IM = 9
        EPERP_REAL = 10
        EPERP_IM = 11
        WFLOW_REAL = 12 
        WFLOW_IM = 13
        BPAR_REAL = 14
        BPAR_IM = 15

    def __init__(self, path : str, poten_timestep = -1):
        with h5py.File(path) as file:
            self.nx = int(file["input/grid/n_x_grid"][()][0])                        # number of grid points in psi (radial) direction
            self.ns = int(file["input/grid/n_s_grid"][()][0])                        # number of grid points in s (poloidal/toroidal) direction
            self.x = file["geom/eps"][()]                                            # radial hamada coordinate psi; shape (NX,)
            self.s = file["geom/s_grid"][()]                                         # poloidal/toroidal hamada coordinate s; shape (NS,)
            self.r_n_flat = file["geom/R"][()]                                         # R(psi,s) flattened as (NX*NS,) array
            self.r_ref = float(file["geom/Rref"][()][0])
            self.z_flat = file["geom/Z"][()]                                         # Z(psi,s) flattened as (NX*NS,) array

            # TODO: only load if needed
            self.g_flat = file["geom/gmap"][()]                                      # gmap(psi,s); shape (NX*NS,)
            self.n_spacing = int(file["input/mode/n_spacing"][()][0])                # TODO
            self.n_mod = int(np.array(file['input/grid/nmod'])[()][0])               # TODO: np.array neccessary?
            #self.rhostar = file["input/spcgeneral/rhostar"][()][0]                  # normed Larmor radius

            self.geom_type = file["input/geom/geom_type"][()][0].decode('utf-8')     # geometry type
            
            sign_b = file["input/geom/signB"][()][0]    # TODO
            sign_j = file["input/geom/signJ"][()][0]    # TODO
            q = file["geom/q"][()]                      # safety factor
            self.q = sign_b * sign_j * np.abs(q)        # safety factor q(psi) with correct sign; shape(NX,)      

            self.non_lin = str(file["input/control/non_linear"][()][0].decode('utf-8'))   # non linear string
            self.is_lin : bool = self.non_lin== 'F'

            # load additional data depending on linear or nonlinear data
            if not self.is_lin:  
                # data is nonlinear
                self.mphi = int(file["diagnostic/diagnos_grid/mphi"][()][0])            # number of grid points in Zeta direction

                poten_keys = [key for key in file["diagnostic/diagnos_fields"].keys() if key.startswith("Poten")]

                if poten_timestep is not None:
                    poten_timestep -= 1     # adjust for index 0
                    if 0 <= poten_timestep<= len(poten_keys):# TODO
                        self.dataset_name = poten_keys[poten_timestep]
                else:   # no timestep given
                    # pick last data set, as it gives the most accurate results
                    self.dataset_name = poten_keys[-1]   

                self.pot3d = file[f"diagnostic/diagnos_fields/{self.dataset_name}"][()]

            else:
                # data is linear
                par_flat = file["diagnostic/diagnos_mode_struct/parallel"][()]
                self.n_sp : int = int(file['input/grid/number_of_species'][0])       # number of species
        # close file

        # reshape parallel data for easier array access
        if self.is_lin:
            self.par = self._reshape_parallel_dat(par_flat, self.n_sp, self.n_mod, self.nx, self.ns)
            self.fcoeffs = self._get_fourier_coeffs(self.par)

        self.r_n = np.reshape(self.r_n_flat, (self.nx, self.ns))
        self.z = np.reshape(self.z_flat, (self.nx, self.ns))
        self.g = np.reshape(self.g_flat, (self.nx, self.ns))

    
    def _reshape_parallel_dat(self, parallel_dat_flat, n_sp, n_mod, nx, ns):
        """
	    Reshapes the parallel data from the `gkwdata.h5` file based on the given parameters for easier access.

        Parameters
        ----------
        parallel_dat_flat : ndarray, shape (n_col*n_sp*n_mod*nx*ns) 
            The flattened parallel data as retrieved from the `gkwdata.h5`.
        n_sp : int
            The number of species.
        n_mod : int
            The number of modes.
        nx : int
            The number of grid points in psi-direction.
        ns : int
            The number of grid points in s-direction.

        Returns
        -------
	    parallel_dat : ndarray, shape (n_col, n_sp, n_mod, nx, ns)
            The reshaped parallel data.
	    """
        n_col : int = len(GKWData.ParallelDatQuantities)                # number of columns (quantities) in parallel.dat for first dimension
    
        set_len = n_sp * n_mod * nx * ns								# the length of one parallel.dat set
        n_parallel = int(parallel_dat_flat.shape[1]/(set_len)) 	  		# get number of parallel.dat sets in current hdf5-file
        start_idx = int((n_parallel-1)*(set_len))
        end_idx = parallel_dat_flat.shape[1]   

        return np.reshape(parallel_dat_flat[:, start_idx : end_idx], shape=(n_col, n_sp, n_mod, nx, ns))

    
    def _get_fourier_coeffs(self, parallel_dat, i_sp = 0, i_mod = 0):
        """
        Retrieves the complex Fourier coefficients from the parallel data in the given file.  

        Parameters
        ----------
        parallel_dat : ndarray of shape (n_col, n_sp, n_mod, nx, ns) 
            The `parallel.dat`. Use the helper function :func:`reshape_parallel_dat` to achieve this.
        i_sp : int
            The species index.
        i_mod : int
            The mode index.    

        Returns
        -------
        fcoeffs: ndarray, shape (nx, ns) 
            The complex fourier coefficients.
        """

        # parallel.dat quantity to select
        real_idx : int = GKWData.ParallelDatQuantities.PHI_REAL.col()
        imag_idx : int = GKWData.ParallelDatQuantities.PHI_IM.col()

        fcoeff_real = parallel_dat[real_idx, i_sp, i_mod, :, :]
        fcoeff_im = parallel_dat[imag_idx, i_sp, i_mod, :, :]

        return fcoeff_real + 1j * fcoeff_im

class ToPoVisData:
    def __init__(self, phi, poten_timestep, method, fx, fs, interpolator):
        self.phi = phi
        self.poten_timestep = poten_timestep
        self.method = method
        self.fx = fx
        self.fs = fs
        self.interpolator = interpolator

    def set_plot_args(self, triang_method, levels, omit_axes, dpi, plot_grid=False):
        self.triang_method = triang_method
        self.levels = levels
        self.omit_axes = omit_axes
        self.dpi = dpi
        self.plot_grid = plot_grid
    
    def save_results(self, x, s, r, z, pot, zeta_s, fcoeffs = None):
        self.nx = len(x)
        self.ns = len(s)
        self.x = x
        self.s = s
        self.r_flat = r
        self.r = np.reshape(r, (self.nx, self.ns))
        self.z_flat = z
        self.z = np.reshape(z, (self.nx, self.ns))
        self.pot = pot
        self.zeta_s = zeta_s
        self.fcoeffs = fcoeffs

    def plot(self, fig=None, ax=None, **kwargs):
        r = kwargs.pop('r', self.r)
        z = kwargs.pop('z', self.z)
        pot = kwargs.pop('pot', self.pot)
        triang_method = kwargs.pop('triang_method', self.triang_method)
        omit_axes = kwargs.pop('omit_axes', self.omit_axes)
        omit_cbar = kwargs.pop('omit_cbar', False)
        plot_grid = kwargs.pop('plot_grid', self.plot_grid)

        return plot(r, z, pot, fig, ax, triang_method, omit_axes, omit_cbar, plot_grid, **kwargs)

################################################## MAIN ##################################################

def main(args = None):
    

    # ---------------------- SETUP PARSER ------------------------

    args = parse_args(args)

    # Argument Validation

    if args.verbose > 3: 
        LOGGING_LEVEL = 3
    else: 
        LOGGING_LEVEL = {3 : logging.DEBUG, 2 : logging.INFO, 1: logging.WARNING, 0: logging.ERROR}[args.verbose]

    logging.basicConfig(level=LOGGING_LEVEL, handlers=[logging.StreamHandler()], force=False)


    HDF5_PATH = str(args.hdf5_filepath)

    POTEN_TIMESTEP : int | None = args.poten_timestep

    PHI = float(args.phi) % (2*np.pi)

    LEGACY_GMAP = args.legacy_gmap

    if int(args.fx) < 1 or int(args.fs) < 1:
        logging.fatal('Interpolating factors FX and FS must be positive integers, exiting.')
        sys.exit(1)

    FX = int(args.fx)
    FS = int(args.fs)

    INTERPOLATE : bool = FX > 1 or FS > 1
    INTERPOLATOR = args.interpolator    

    VALID_METHODS = ['nearest', 'linear', 'cubic', 'quintic']
    if args.method in VALID_METHODS:
        METHOD = str(args.method)
        ORDER = {'nearest': 0, 'linear': 1, 'cubic': 3, 'quintic': 5}[METHOD]
    else:
        logging.fatal(f"No valid interpolation method specified. Got {args.method}. Valid options are {str(VALID_METHODS)}. Exiting.")
        sys.exit(1)  

    ZONAL = args.zonal

    if args.data_out is not None:
        DATA_OUT : str = str(args.data_out).strip()
    else:
        DATA_OUT = None

    # Set plot args

    TRIANG_METHOD = args.triang_method
    LEVELS = int(args.levels)
    DPI = int(args.dpi)

    if args.plot_out is not None:
        PLOT_OUT : str = str(args.plot_out).strip()
    else:
        PLOT_OUT = None

    OMIT_AXES = bool(args.omit_axes)
    PLOT_GRID = bool(args.plot_grid)

    DSF = int(args.dsf)

    out = ToPoVisData(PHI, POTEN_TIMESTEP, METHOD, FX, FS, INTERPOLATOR)
    out.set_plot_args(TRIANG_METHOD, LEVELS, OMIT_AXES, DPI, PLOT_GRID)

    # ----------------------------------------- GKWDATA ----------------------------------------------

    logging.info(f'Reading file {HDF5_PATH}')
    dat = GKWData(HDF5_PATH, poten_timestep=POTEN_TIMESTEP)

    # --------------------------------------- PREPERATION --------------------------------------------

    # Downsample for debugging
    if DSF < 0:
        logging.fatal(f"Downsample factor must be positive. Exiting.")
        sys.exit(1)
    elif DSF == 0:
        pass
    elif DSF == 1:
        logging.warning(f"Got DSF of 1. Ignoring.")
        pass
    elif DSF > 1:
        logging.warning(f"Downsampling grid by factor {DSF}")
        slc = np.s_[::int(DSF)]
        dat.s = dat.s[slc]
        dat.ns = len(dat.s)
        dat.g = dat.g[:, slc]
        dat.g_flat = np.ravel(dat.g, 'C')
        dat.r_n = dat.r_n[:, slc]
        dat.r_n_flat = np.ravel(dat.r_n, 'C')
        dat.z = dat.z[:, slc]
        dat.z_flat = np.ravel(dat.z, 'C')
        
        try:
            dat.fcoeffs = dat.fcoeffs[:, slc]
        except AttributeError:
            pass
            
        try:
            dat.pot3d = dat.pot3d[slc, :, :]
        except AttributeError:
            pass
    

    IS_LIN = dat.is_lin

    if LEGACY_GMAP:
        GEOM = dat.geom_type
    else:
        GEOM = 'universal'
    
    OVERLAP = 4

    # perform zeta-shift
    logging.info(f'Performing zeta-shift')
    zeta_s = shift_zeta(dat.g, dat.s, PHI, GEOM,
                    q=dat.q, 
                    r_n=dat.r_n,
                    r_ref=dat.r_ref, 
                    nx=dat.nx, 
                    ns=dat.ns)
    # shape (nx,ns)
    zeta_s_flat = np.ravel(zeta_s)

    # --------------------- PREPARE GRID ---------------------

    if INTERPOLATE:
        # CREATE FINE HAMADA GRID
        x_fine, s_fine = interpolate_hamada_grid(dat.x, dat.s, FX, FS, extrapolate_s=True)
        
        nx_fine = len(x_fine)
        ns_fine = len(s_fine)
        # precalculate different coordinate representations for reuse
        xs_fine = xx_fine, ss_fine = np.meshgrid(x_fine, s_fine, indexing='ij')   # ! has to use indexing='ij'
        xs_points_fine = grid_to_points(xs_fine)

        # extend s-grid (not wrap!)
        s = extend_regular_array(dat.s, OVERLAP)     # shape (ns+2n)
        x = dat.x                                    # shape (nx)
        r_n = extend_periodically(dat.r_n, OVERLAP, axis=1)
        z = extend_periodically(dat.z, OVERLAP, axis=1)

        # NOTE: The poloidal coordinates ALWAYS get interpolated using RGI

        logging.info(f'Interpolating poloidal grid')
        r_rgi = scipy.interpolate.RegularGridInterpolator((x, s), r_n, method=METHOD)
        r_n_fine_flat = r_rgi(xs_points_fine)
        r_n_fine = np.reshape(r_n_fine_flat, shape=(nx_fine, ns_fine))

        z_rgi = scipy.interpolate.RegularGridInterpolator((x,s), z, method=METHOD)
        z_fine_flat = z_rgi(xs_points_fine)
        z_fine = np.reshape(z_fine_flat, shape=(nx_fine, ns_fine))

        rz_points_fine = np.column_stack((r_n_fine_flat, z_fine_flat))

    # --------------------- CALCULATE POT ---------------------

    if IS_LIN: # LINEAR SIMULATION

        zeta_s = shift_zeta(dat.g, dat.s, PHI, GEOM,
                            q=dat.q, 
                            r_n=dat.r_n,
                            r_ref=dat.r_ref, 
                            nx=dat.nx, 
                            ns=dat.ns)
        # shape (nx,ns)
        zeta_s_flat = np.ravel(zeta_s)

        fcoeffs = dat.fcoeffs
        fcoeffs_flat = np.ravel(fcoeffs)

        # wave vector
        k = 2 * np.pi * dat.n_mod * dat.n_spacing   # constant

        if INTERPOLATE:
            # INTERPOLATE DATA

            if INTERPOLATOR == 'rgi':
                # APPLY PERIODIC BOUNDARY CONDITIONS

                # extend s-grid (not wrap!)
                s = extend_regular_array(dat.s, OVERLAP)     # shape (ns+2n)
                x = dat.x    # shape (nx)

                # ζ(s) = ζ(s±1) ∓ q
                zeta_s = extend_periodically(zeta_s, OVERLAP, 1)    # extend grid periodically in s
                zeta_s[:, :OVERLAP] -= dat.q[:, None]               # apply boundary condition for zeta[s < -0.5]
                zeta_s[:, -OVERLAP:None] += dat.q[:, None]          # apply boundary condition for zeta[s > 0.5]
                # zeta_s = zeta_s % 1                               # map zeta back to [0,1]

                # f(s) = f(s±1) * exp(±ikq)
                fcoeffs = extend_periodically(fcoeffs, OVERLAP, 1)              # extend grid periodically in s
                fcoeffs[:, :OVERLAP] *= np.exp(1j * k * dat.q[:, None])
                fcoeffs[:, -OVERLAP:None] *= np.exp(-1j * k * dat.q[:, None])

                # interpolate zeta-shift
                logging.info(f'Interpolating zeta-shift')
                zeta_s_rgi = scipy.interpolate.RegularGridInterpolator((x, s), zeta_s, method=METHOD)
                zeta_s_fine_flat = zeta_s_rgi(xs_points_fine)   # shape (nx_fine*ns_fine,)

                # interpolate complex fourier coefficients
                logging.info(f'Interpolating fcoeffs')
                fcoeffs_hgi = scipy.interpolate.RegularGridInterpolator((x, s), fcoeffs, method=METHOD)
                fcoeffs_fine_flat = fcoeffs_hgi(xs_points_fine)   # shape (nx_fine*ns_fine,)
            elif INTERPOLATOR == 'rbfi':
                """
                This interpolates poloidally using radial basis functions.

                However, as we define points only by their position on the poloidal slice (R,Z), it's not possible to apply parallel boundary conditions. To do that, we would need to add a new sets of points at the same poloidal position but with different (shifted) values. Due to the functionality of the RBF interpolation this is not possible. 
                
                For this reason, regular grid interpolation in hamada coordinates is almost always the preffered option and leads to better results. This is being kept in code mainly for demonstration purposes.
                """

                rz_points = np.column_stack((dat.r_n_flat, dat.z_flat))

                # FIXME: add to argparse
                rbf_kwargs = {'neighbors':200}

                # interpolate zeta-shift
                logging.info(f'Interpolating zeta-shift')
                zeta_s_rbfi = scipy.interpolate.RBFInterpolator(rz_points, zeta_s_flat, **rbf_kwargs)
                zeta_s_fine_flat = zeta_s_rbfi(rz_points_fine)   # shape (nx_fine*ns_fine,)

                # interpolate complex fourier coefficients
                logging.info(f'Interpolating fcoeffs')
                fcoeffs_rbfi = scipy.interpolate.RBFInterpolator(rz_points, fcoeffs_flat, **rbf_kwargs)
                fcoeffs_fine_flat = fcoeffs_rbfi(rz_points_fine)   # shape (nx_fine*ns_fine,)

            logging.info('Calculating potential')
            pot_fine_flat = calculate_potential(fcoeffs_fine_flat, zeta_s_fine_flat, k)

            out.save_results(x_fine, s_fine, r_n_fine_flat, z_fine_flat, pot_fine_flat, zeta_s_fine_flat, fcoeffs_fine_flat)
        else:
            logging.info('Calculating potential')
            pot = calculate_potential(fcoeffs, zeta_s, k)
            pot_flat = np.ravel(pot)
            zeta_s_flat = np.ravel(zeta_s)
            fcoeffs_flat = np.ravel(dat.fcoeffs)

            out.save_results(dat.x, dat.s, dat.r_n_flat, dat.z_flat, pot_flat, zeta_s_flat, fcoeffs_flat)          


    else: # --------------------- NON LINEAR SIMULATION ---------------------
        logging.info('Nonlinear simulation')

        # NOTE: maybe it's smarter to first interpolate on the 1/n_spacing part of the torus and THEN repeat periodically.
        # This could lead to derivatives being non-continuous at the edges, but avoids redundant intense calculations especially for higher orders of interpolations.
        whole_zeta = make_zeta_grid(dat.mphi, dat.n_spacing)    # shape (mphi*n_spacing)

        pot3d = dat.pot3d

        if not ZONAL:
            pot_mean = np.mean(pot3d, axis=(0,2))    # shape (nx,)
            pot3d = pot3d - pot_mean[None, :, None]

        pot = repeat_pot(pot3d, dat.n_spacing)        # shape (ns, nx, mphi*n_spacing)

        s = dat.s
        x = dat.x
        z = whole_zeta

        if INTERPOLATE:
            # extend grid
            s_e = extend_regular_array(s, OVERLAP)
            x_e = x
            z_e = z

            # extend grid periodically
            s_p = extend_periodically(s, OVERLAP, 0)
            x_p = x
            z_p = z

            sss_p, xxx_p, zzz_p = np.meshgrid(s_p, x_p, z_p, indexing='ij')

            # ζ(s) = ζ(s±1) ∓ q
            zzz_p[-OVERLAP:None, :, :] -= dat.q[None, :, None] 
            zzz_p[:OVERLAP, :, :] += dat.q[None, :, None] 
            zzz_p = zzz_p % np.max(z) # FIXME: this is inaccurate. max(zeta) is smaller 1

            # sxz_p = sss_p, xxx_p, zzz_p

            # interpolate pot
            logging.info('Constructing splines, this might take a while...')
            pot3d_rgi = scipy.interpolate.RegularGridInterpolator((s, x, z), pot, method=METHOD)

            pot3d_p = np.zeros((len(s_e), len(x_e), len(z_e)))

            # TODO: maybe this can be done in one step?
            
            # copy original pot
            pot3d_p[OVERLAP:-OVERLAP, :, :] = pot

            slc = np.s_[-OVERLAP:None, :, :]
            points = grid_to_points((sss_p[slc], xxx_p[slc], zzz_p[slc]))
            p = pot3d_rgi(points)
            pot3d_p[slc] = np.reshape(p, shape=(np.shape(pot3d_p[slc])))

            slc = np.s_[:OVERLAP, :, :]
            points = grid_to_points((sss_p[slc], xxx_p[slc], zzz_p[slc]))
            p = pot3d_rgi(points)
            pot3d_p[slc] = np.reshape(p, shape=(np.shape(pot3d_p[slc])))

            # apply parallel boundary condition to zeta shift

            # ζ(s) = ζ(s±1) ∓ q
            zeta_s_p = extend_periodically(zeta_s, OVERLAP, 1)    # extend grid periodically in s
            zeta_s_p[:, :OVERLAP] -= dat.q[:, None]               # apply boundary condition for zeta[s < -0.5]
            zeta_s_p[:, -OVERLAP:None] += dat.q[:, None]          # apply boundary condition for zeta[s > 0.5]
                
            # interpolate zeta-shift
            logging.info(f'Interpolating zeta-shift')
            zeta_s_rgi = scipy.interpolate.RegularGridInterpolator((x_e, s_e), zeta_s_p, method=METHOD)
            zeta_s_fine_flat = zeta_s_rgi(xs_points_fine) 
            zeta_s_fine_flat = zeta_s_fine_flat % np.max(z_e)     # FIXME: this is inaccurate!
            zeta_s_fine = np.reshape(zeta_s_fine_flat, (nx_fine, ns_fine))

            sss_fine = np.expand_dims(ss_fine, -1)
            xxx_fine = np.expand_dims(xx_fine, -1)
            zzzeta_s_fine = np.expand_dims(zeta_s_fine, -1)

            sxz_fine = sss_fine, xxx_fine, zzzeta_s_fine
            sxz_fine_points = grid_to_points(sxz_fine)

            logging.info(f"Interpolating 3d potential. This might take a while...")
            pot3d_p_rgi = scipy.interpolate.RegularGridInterpolator((s_e, x_e, z_e), pot3d_p, method=METHOD)
            pot3d_fine_flat = pot3d_p_rgi(sxz_fine_points)
            pot3d_fine = np.reshape(pot3d_fine_flat, np.shape(sss_fine))
            pot_fine = pot3d_fine[:,:,0]
            
            pot_fine_flat = np.ravel(pot_fine, 'C') 
            # NOTE: this is set to 'C' to transpose the potential from (s,x) to (x,s)

            out.save_results(x_fine, s_fine, r_n_fine_flat, z_fine_flat, pot_fine_flat, zeta_s_fine_flat)
        else:
            logging.info("Interpolating potential on zeta-shift.")
            pot_ev = np.zeros(shape=(dat.nx, dat.ns))
            for i in range(dat.nx):
                for j in range(dat.ns):
                    spl = scipy.interpolate.splrep(z, pot[j,i,:], k=3)
                    # NOTE: mapping to max(z) is inaccurate
                    pot_ev[i,j] = scipy.interpolate.splev(zeta_s[i,j] % np.max(z), spl)
            
            pot_flat = np.ravel(pot_ev, order='C')

            out.save_results(x, s, dat.r_n_flat, dat.z_flat, pot_flat, zeta_s)


    # -------------------------------------- SAVING RESULTS TO FILE -----------------------------------------

    # FIXME: move this to ToPoVisData
    if DATA_OUT:
        logging.info(f'Saving data to file: {DATA_OUT}')
        with h5py.File(DATA_OUT, 'w') as f:
            # TODO: categorize these in subset structure
            # save params
            f.create_dataset("order",data=ORDER)
            f.create_dataset("phi", dtype='f', data=PHI)
            f.create_dataset("poten_timestep", dtype='i', data=POTEN_TIMESTEP)
            f.create_dataset("fx", dtype='i', data=FX)
            f.create_dataset("fs", dtype='i', data=FS)

            # save data
            f.create_dataset("x", dtype='f', data=out.x)
            f.create_dataset("s", dtype='f', data=out.s)
            f.create_dataset("r_n", dtype='f', data=out.r_flat)
            f.create_dataset("z", dtype='f', data=out.z_flat)
            f.create_dataset("pot", dtype='f', data=out.pot)
            f.create_dataset("zeta_s", dtype='f', data=out.zeta_s)

            if dat.is_lin:
                f.create_dataset("fcoeffs_real", dtype='f', data=out.fcoeffs.real)
                f.create_dataset("fcoeffs_imag", dtype='f', data=out.fcoeffs.imag)


    # --------------------------------------------- PLOTTING ---------------------------------------------

    if PLOT_OUT:
        logging.info(f'Preparing plot')
        fig, ax = out.plot()

        logging.info(f'Saving plot to {PLOT_OUT}')
        fig.savefig(PLOT_OUT, dpi=DPI)

    
    logging.info('Done')
    return out

if __name__ == "__main__":
    main(sys.argv[1:])