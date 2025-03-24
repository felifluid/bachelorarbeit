################################################### IMPORTS ##################################################

import logging
import argparse
import sys
import numpy as np
import h5py
import enum
import scipy.integrate
import scipy.interpolate
import matplotlib.tri
import matplotlib.pyplot as plt
import pathlib

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

def periodic_wrap(a, overlap : int, axis : int | list[int]) -> np.ndarray:
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
    
def shift_zeta(g, s, phi, geom_type, sign_b=None, sign_j=None, q=None, r_n=None, r_ref=None, nx=None, ns=None):
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
    if geom_type == 'circ':
        zeta = -phi / (2*np.pi) + g
    elif geom_type == 's-alpha':
        #FIXME: equation for s-alpha is not right
        zeta = (sign_b * sign_j) / (2*np.pi) * (2*np.pi * np.abs(q) * s - phi)
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

    return np.repeat(pot, n_spacing, axis=2)
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

def my_tricontourf(ax, triangulation, z, show_grid=False, **kwargs):
    x = triangulation.x
    y = triangulation.y
    cmap = kwargs.pop('cmap', 'seismic')
    levels = kwargs.pop('levels', 100)

    ax.tricontourf(triangulation, z, levels=levels, zorder=0, cmap=cmap, **kwargs)
    if show_grid : plot_grid(ax, triangulation, z, **kwargs)

def plot_grid(ax, triangulation, z, **kwargs):
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
                        default=-1, 
                        help='(optional) Time step for potential. If not specified the last timestep is being used.')
    parser.add_argument('-z', '--zonal',
                        action='store_true',
                        dest='zonal',
                        help="(optional) If set plots zonal potential. This is only being used in nonlinear simulations.")
    plot_group = parser.add_argument_group('plot', description='Parameters for the plot')
    plot_group.add_argument('--triang-method', 
                            type=str,
                            choices=('regular', 'delaunay'),
                            dest='triang_method',
                            default='regular', 
                            help="(optional) Method to use for triangulation. 'regular' creates regular triangles in hamada coordinates, whereas 'delaunay' performs a Delaunay triangulation in poloidal coordinates. Default is 'regular'.")
    plot_group.add_argument('--plot-hamada', 
                            action='store_true',
                            dest='plot_hamada', 
                            help='(optional) If set, plots in Hamada coordinates instead of poloidal coordinates.')
    plot_group.add_argument('--plot-out',
                            dest='plot_out',
                            type=str,
                            default='plot.pdf',
                            help="(optional) Specify a file to write the plot into WITH extension. Can either be a full path, or a filename. If it's a filename, it will save to the current directory. Default is 'plot.pdf'."
                            )
    plot_group.add_argument('--plot-grid',
                            dest='plot_grid',
                            action='store_true',
                            help='(optional) plots scatter plot with triangulation instead of a contour plot. Good for debugging and spotting triangulation artifacts.'
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
    interpolation_group = parser.add_argument_group('interpolation', description='Set interpolation parameters.')
    interpolation_group.add_argument('fx', 
                                     type=int, 
                                     help='Factor by which to refine the psi-grid through interpolation.')
    interpolation_group.add_argument('fs', 
                                     type=int, 
                                     help='Factor by which to refine the s-grid through interpolation.')
    interpolation_group.add_argument('--interpolator',
                                     dest='interpolator',
                                     choices=('rgi', 'rbfi'),
                                     default='rgi',
                                     help="(optional) Which interpolator to use to interpolate the potential. 'rgi' uses the RegularGridInterpolator, which interpolates on hamada coordinates, 'rbfi' uses the RBFInterpolator, which interpolates in poloidal coordinates. Default is 'rgi'.")
    interpolation_group.add_argument('--periodic',
                                     action='store_true',
                                     help='If supplied, applies period boundary condition to grid to interpolate between s=-0.5 and s=0.5. Has no effect when fs=1.'
                                    )
    interpolation_group.add_argument('--method',
                                     metavar='method',
                                     choices=('nearest', 'linear', 'cubic', 'quintic'),
                                     default='quintic',
                                     help="(optional) Method of interpolation. Valid options are: 'nearest', 'linear', 'cubic', 'quintic'. Default is 'quintic'. When faced with memory or processing constraints this should be changed.")
    interpolation_group.add_argument('--data-out',
                                     dest='data_out',
                                     type=str, 
                                     help="(optional) Specify a h5-file to write the interpolation data into. Can either be a full path, or a filename. If it's a filename, it will save to the current directory. Default is 'topovis_data.h5'.")

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
            self.q = file["geom/q"][()]                                              # safety factor q(psi); shape (NX,)
            self.g_flat = file["geom/gmap"][()]                                      # gmap(psi,s); shape (NX*NS,)
            self.n_spacing = int(file["input/mode/n_spacing"][()][0])                # TODO
            self.n_mod = int(np.array(file['input/grid/nmod'])[()][0])               # TODO: np.array neccessary?
            #self.rhostar = file["input/spcgeneral/rhostar"][()][0]                  # normed Larmor radius
            self.sign_b = file["input/geom/signB"][()][0]                            # TODO
            self.sign_j = file["input/geom/signJ"][()][0]                            # TODO
            self.geom_type = file["input/geom/geom_type"][()][0].decode('utf-8')     # geometry type

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

    POTEN_TIMESTEP = int(args.poten_timestep)

    PHI = float(args.phi) % (2*np.pi)

    FX = int(args.fx)
    FS = int(args.fs)

    INTERPOLATE : bool = FX > 1 or FS > 1
    PERIODIC : bool = args.periodic
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

    PLOT_HAMADA = args.plot_hamada
    TRIANG_METHOD = args.triang_method
    LEVELS = int(args.levels)
    DPI = int(args.dpi)
    PLOT_OUT = str(args.plot_out)
    OMIT_AXES = bool(args.omit_axes)
    PLOT_GRID = bool(args.plot_grid)

    # ---------------------------------------------- GKWDATA -------------------------------------------------

    logging.info(f'Reading file {HDF5_PATH}')
    dat = GKWData(HDF5_PATH, poten_timestep=POTEN_TIMESTEP)

    # perform zeta-shift
    logging.info(f'Performing zeta-shift')
    zeta_s = shift_zeta(dat.g, dat.s, PHI, dat.geom_type, 
                    sign_b=dat.sign_b, 
                    sign_j=dat.sign_j, 
                    q=dat.q, 
                    r_n=dat.r_n,
                    r_ref=dat.r_ref, 
                    nx=dat.nx, 
                    ns=dat.ns)
    # shape (nx,ns)

    if dat.is_lin:
        fcoeffs = dat.fcoeffs    # shape (nx,ns)

    if PERIODIC and FS > 1:     # apply periodic boundary condition
        """
        Extend all variables depending on s periodically using specific periodic boundary conditions.
        This allows interpolation between first and last dataset.
        """
        logging.info("Applying double periodic boundary condition")
        
        # number of grid points to extend in each direction (overlap)
        n = 4

        # extend the s-grid out of bounds without wrapping periodically 
        # as `RegularGridInterpolator` needs a regular spaced, strictly ascending or descending grid without discontinuities
        s = extend_regular_array(dat.s, n)     # shape (ns+2n)
        x = dat.x                              # shape (nx)

        ns = len(s)   # ns+2n
        nx = len(x)   # nx

        if INTERPOLATOR == 'rgi':
            # z(s0) = z(s0 ± 1) ± q = z(s1) ± q
            zeta_s = periodic_wrap(zeta_s, n, 1)   # extend grid periodically in s
            zeta_s[:, :n] += dat.q[:, None]        # apply boundary condition for zeta[s < -0.5]
            zeta_s[:, -n:None] -= dat.q[:, None]   # apply boundary condition for zeta[s > 0.5]
            zeta_s = zeta_s % 1                    # map zeta back to [0,1]

            if dat.is_lin:
                # fourier coefficients only used in linear simulations

                fcoeffs = periodic_wrap(fcoeffs, n, 1)  # extend grid periodically in s
                # ff[:, :n] *= np.exp(1j * k * dat.q[:, None])
                # ff[:, -n:None] *= np.exp(-1j * k * dat.q[:, None])

        r_n = periodic_wrap(dat.r_n, overlap=n, axis=1)
        z = periodic_wrap(dat.z, overlap=n, axis=1)
    else:
        # don't apply periodic boundary condition
        x = dat.x
        s = dat.s

        nx = dat.nx
        ns = dat.ns

        r_n = dat.r_n
        z = dat.z

    # ------------------- INTERPOLATE GRID ------------------------

    if INTERPOLATE:
        # Interpolate Grid
        logging.info('Interpolating hamada grid')
        x_fine, s_fine = interpolate_hamada_grid(dat.x, dat.s, FX, FS, extrapolate_s=PERIODIC)

        # new grid-size
        nx_fine = len(x_fine)
        ns_fine = len(s_fine)

        logging.info(f'Fine grid resolution: nx={nx_fine}, ns={ns_fine}')

        # precalculate different coordinate representations for reuse
        xs_fine = xx_fine, ss_fine = np.meshgrid(x_fine, s_fine, indexing='ij')   # ! has to be in this order, or use indexing='ij'
        xs_points_fine = grid_to_points(xs_fine)

        logging.info(f'Interpolating poloidal grid')
        r_rgi = scipy.interpolate.RegularGridInterpolator((x,s), r_n, method=METHOD)
        r_n_fine_flat = r_rgi(xs_points_fine)
        r_n_fine = np.reshape(r_n_fine_flat, shape=(nx_fine, ns_fine))

        z_rgi = scipy.interpolate.RegularGridInterpolator((x,s), z, method=METHOD)
        z_fine_flat = z_rgi(xs_points_fine)
        z_fine = np.reshape(z_fine_flat, shape=(nx_fine, ns_fine))
    else: # no interpolation
        # use sparse grid as "fine" grid
        x_fine = x
        s_fine = s
        ns_fine = ns
        nx_fine = nx

        r_n_fine = dat.r_n
        r_n_fine_flat = dat.r_n_flat
        z_fine = dat.z
        z_fine_flat = dat.z_flat

    # --------------------------------------- CALCULATE POTENTIAL ------------------------------------------

    if dat.is_lin:  # linear simulation

            # wave vector
            k = 2 * np.pi * dat.n_mod * dat.n_spacing   # constant

            if INTERPOLATE:
                if INTERPOLATOR == 'rgi':
                    logging.info(f'Interpolation geometry: Hamada')

                    # interpolate zeta-shift
                    logging.info(f'Interpolating zeta-shift')
                    zeta_s_rgi = scipy.interpolate.RegularGridInterpolator((x, s), zeta_s, method=METHOD)
                    zeta_s_fine_flat = zeta_s_rgi(xs_points_fine)   # shape (nx_fine*ns_fine,)

                    # interpolate complex fourier coefficients
                    logging.info(f'Interpolating fcoeffs')
                    fcoeffs_hgi = scipy.interpolate.RegularGridInterpolator((x, s), fcoeffs, method=METHOD)
                    fcoeffs_fine_flat = fcoeffs_hgi(xs_points_fine)   # shape (nx_fine*ns_fine,)

                    pot_fine_flat = calculate_potential(fcoeffs_fine_flat, zeta_s_fine_flat, k)
                elif INTERPOLATOR == 'rbfi':
                    logging.info(f'Interpolation geometry: Poloidal')

                    # calculate potential on sparse grid
                    pot = calculate_potential(fcoeffs, zeta_s, k)

                    rz_points = np.column_stack((np.ravel(dat.r_n), np.ravel(dat.z)))
                    rz_points_fine = np.column_stack((r_n_fine_flat, z_fine_flat))

                    # FIXME: add to argparse
                    rbf_kwargs = {'neighbors':150, 'kernel': 'cubic', 'degree': 1}

                    pot_rbfi = scipy.interpolate.RBFInterpolator(rz_points, np.ravel(pot), **rbf_kwargs)
                    pot_fine_flat = pot_rbfi(rz_points_fine)

            else:   # do not interpolate
                # use sparse values as "fine" values
                fcoeffs_fine_flat = np.ravel(fcoeffs)
                zeta_s_fine_flat = np.ravel(zeta_s)

                pot_fine_flat = calculate_potential(fcoeffs_fine_flat, zeta_s_fine_flat, k)
            

    else:   # nonlinear simulation
        logging.info('Nonlinear simulation')

        # NOTE: maybe it's smarter to first interpolate on the 1/n_spacing part of the torus and THEN repeat periodically.
        # This could lead to derivatives being non-continuous at the edges, but avoids redundant intense calculations especially for higher orders of interpolations.
        whole_zeta = make_zeta_grid(dat.mphi, dat.n_spacing)    # shape (mphi*n_spacing)

        pot3d = dat.pot3d

        if not ZONAL:
            pot_mean = np.mean(pot3d, axis=(0,2))    # shape (nx,)
            pot3d = pot3d - pot_mean[None, :, None]

        whole_pot = repeat_pot(pot3d, dat.n_spacing)        # shape (ns, nx, mphi*n_spacing)

        logging.info("Interpolating potential")

        # Extend grid out of bounds

        n = 4   # overlap

        logging.info('Extending grid out of bounds')
        """
        use periodic boundary condition to extend 3d potential to > 0.5 and < -0.5

        let s0=0.5+Δs/2
        f(s0, x, z0) = f(s0-1, x, z0-q)
        we define s1=s0-1=-0.5+Δs/2
        """

        s0 = extend_regular_array(dat.s, n)
        x0 = dat.x
        z0 = extend_regular_array(whole_zeta, n)

        sxz0 = ss0, xx0, zz0 = np.meshgrid(s0, x0, z0, indexing='ij')

        ns0 = len(s0)
        nx0 = len(x0)
        nz0 = len(z0)

        logging.info('Applying periodic boundary conditions to grid')
        # s1[-0.5 < s < 0.5] = s0
        ss1 = np.copy(ss0)
        # s1[s > 0.5] = s0 - 1 
        ss1[-n:None, :, :] -= 1
        # s1[s < -0.5] = s0 + 1 
        ss1[:n, :, :] += 1

        # x1 = x0
        xx1 = xx0

        # z1[-0.5 < s < 0.5] = z0
        zz0 = np.copy(zz0)
        # z1[s > 0.5] = z0 - q(x) 
        zz0[-n:None, :, :] -= dat.q[None, :, None]
        # z1[s < -0.5] = z0 + q(x) 
        zz0[:n, :, :] += dat.q[None, :, None]
        # map back to [0,1]
        zz0 = zz0 % 1

        # make s-x-z-grid
        sxz1 = ss1, xx1, zz0
        sxz1_points = grid_to_points(sxz1)

        logging.info('Applying periodic boundary condition to grid')

        # init empty extended pot
        pot1 = np.zeros(shape=(ns0, nx0, nz0), dtype=float)

        # copy original data
        pot1[n:-n, :, n:-n] = whole_pot

        # extend 3d pot periodically in zeta
        pot1[n:-n, :, :n] = array_slice(whole_pot, 2, slice(-n,None))
        pot1[n:-n, :, -n:None] = array_slice(whole_pot, 2, slice(n))

        logging.info('Constructing splines, this might take a while...')
        pot_ex_rgi = scipy.interpolate.RegularGridInterpolator((dat.s, dat.x, z0), pot1[n:-n, :, :], method=METHOD)

        logging.info('Evaluating potential on extended grid.')
        pot1_flat= pot_ex_rgi(sxz1_points)
        pot1 = np.reshape(pot1_flat, shape=(ns0, nx0, nz0))

        logging.info('Creating fine extrapolated grid')
        x_ex_fine, s_ex_fine, (si_lo, si_hi) = interpolate_hamada_grid(x, s, FX, FS, extrapolate_s=True, include_boundaries='hi')
        xs_ex_fine = xx_ex_fine, ss_ex_fine = np.meshgrid(x_ex_fine, s_ex_fine, indexing='ij')
        xs_points_ex_fine = grid_to_points(xs_ex_fine)

        sxz_fine = np.meshgrid(s_ex_fine, x_ex_fine, )

        logging.info('Performing zeta-shift')
        zz_s = shift_zeta(dat.g, dat.s, PHI, dat.geom_type, 
                                sign_b=dat.sign_b, 
                                sign_j=dat.sign_j, 
                                q=dat.q, 
                                r_n=dat.r_n,
                                r_ref=dat.r_ref, 
                                nx=dat.nx, 
                                ns=dat.ns)
        # zz_s is shape (nx, ns)

        logging.info(f'Interpolating zeta-shift')

        # init empty extended pot
        zz_s1 = np.zeros(shape=(nx0, ns0), dtype=float)

        # zeta_s[-0.5 < s < 0.5] = zeta_s
        zz_s1[:, n:-n] = zz_s
        # zeta_s[s > 0.5] = zeta_s - q(x) 
        zz_s1[:, -n:None] =  dat.q[:, None]
        # z1[s < -0.5] = z0 + q(x) 
        zz_s1[:, :n] += dat.q[:, None]


        zeta_s_hgi = scipy.interpolate.RegularGridInterpolator((dat.x, dat.s), zz_s, method=METHOD)
        zeta_s_fine_flat = zeta_s_hgi(xs_points_fine) % 1
        zeta_s_fine = np.reshape(zeta_s_fine_flat, shape=(len(x_fine), len(s_fine)))


        logging.info('Constructing splines on extended potential, this might take a while...')
        pot1_rgi = scipy.interpolate.RegularGridInterpolator((s0, x0, z0), pot1, method=METHOD)

        logging.info('Evaluating potential on zeta-shift')
        pot_fine = pot1_rgi()

        # Compatibilty formatting

        # pot needs to be transposed from (s,x) to (x,s)
        pot_fine_flat = np.ravel(pot_fine.T, order='F')

        x_fine = x_ex_fine
        s_fine = s_ex_fine
        nx_fine = len(x_ex_fine)
        ns_fine = len(s_ex_fine)

        xs_fine = xx_fine, ss_fine = xs_ex_fine
        xs_points_fine = xs_points_ex_fine


    # -------------------------------------- SAVING RESULTS TO FILE -----------------------------------------

    if DATA_OUT:
        logging.info(f'Saving interpolation results to file: {DATA_OUT}')
        with h5py.File(DATA_OUT, 'w') as f:
            # TODO: categorize these in subset structure
            # save params
            f.create_dataset("order",data=ORDER)
            f.create_dataset("phi", dtype='f', data=PHI)
            f.create_dataset("poten_timestep", dtype='i', data=POTEN_TIMESTEP)
            f.create_dataset("fx", dtype='i', data=FX)
            f.create_dataset("fs", dtype='i', data=FS)

            # save data
            f.create_dataset("nx", dtype='i', data=nx_fine)
            f.create_dataset("ns", dtype='i', data=ns_fine)
            f.create_dataset("x", dtype='f', data=x_fine)
            f.create_dataset("s", dtype='f', data=s_fine)
            f.create_dataset("r_n", dtype='f', data=r_n_fine_flat)
            f.create_dataset("z", dtype='f', data=z_fine_flat)
            f.create_dataset("pot", dtype='f', data=pot_fine_flat)
            f.create_dataset("zeta_s", dtype='f', data=zeta_s_fine_flat)
            f.create_dataset("q", dtype='f', data=dat.q)

        if dat.is_lin:
            f.create_dataset("fcoeffs_real", dtype='f', data=fcoeffs_fine_flat.real)
            f.create_dataset("fcoeffs_imag", dtype='f', data=fcoeffs_fine_flat.imag)


    # --------------------------------------------- PLOTTING ---------------------------------------------

    if TRIANG_METHOD == 'regular':
        logging.info('Creating regular triangle grid')
        is_plot_periodic = PERIODIC or FS == 1
        triangles = make_regular_triangles(nx_fine, ns_fine, periodic=is_plot_periodic)
    elif TRIANG_METHOD == 'delaunay':
        logging.info('Performing delaunay triangulation')
        # NOTE: this ALWAYS creates periodic triangles
        # TODO: maybe it's possible to filter these out? then again it's unneccessary when periodic interpolation works
        triangles = matplotlib.tri.Triangulation(r_n_fine_flat, z_fine_flat).triangles
    else:
        logging.fatal(f"No other triangulation method supported other than 'regular' and 'delaunay', got {TRIANG_METHOD}. Exiting")
        sys.exit(1)

    logging.info('Creating plot')

    plot_args = {}
    plot_args['vmin'] = np.min(pot_fine_flat)
    plot_args['vmax'] = np.max(pot_fine_flat)
    plot_args['levels'] = LEVELS

    fig, ax = plt.subplots()

    if OMIT_AXES:
        ax.set_xticks([])
        ax.set_yticks([])

    if PLOT_HAMADA: # hamada coordinates
        # NOTE: this has to be set to order='F'
        xx_fine_flat, ss_fine_flat = np.ravel(xx_fine, order='F'), np.ravel(ss_fine, order='F')
        triangulation = matplotlib.tri.Triangulation(xx_fine_flat, ss_fine_flat, triangles=triangles)
    else:   # polodial coordinates
        triangulation = matplotlib.tri.Triangulation(r_n_fine_flat, z_fine_flat, triangles=triangles)

        # plot closed line at x=0 and x=-1 (inner and outer radial border of data)
        ax.plot(np.append(r_n_fine[0, :], r_n_fine[0,0]), np.append(z_fine[0, :], z_fine[0,0]), ls='-', c=(0,0,0, 0.8), lw=0.2, zorder=11)
        ax.plot(np.append(r_n_fine[-1, :], r_n_fine[-1,0]), np.append(z_fine[-1, :], z_fine[-1,0]), ls='-', c=(0,0,0, 0.8), lw=0.2, zorder=11)

        # fill inner empty circle with white area, this is only neccessary in case of delaunay triangulation
        ax.fill(r_n_fine[0, :], z_fine[0, :], color='white', zorder=10)
        ax.set_aspect('equal')

    if PLOT_GRID:    
        plot_grid(ax, triangulation, pot_fine_flat, **plot_args)
    else:
        my_tricontourf(ax, triangulation, pot_fine_flat, show_grid=False, cmap='seismic', **plot_args)

    logging.info(f'Saving plot to {PLOT_OUT}')
    plt.savefig(PLOT_OUT, dpi=DPI)

    logging.info('Done')

if __name__ == "__main__":
    main(sys.argv[1:])