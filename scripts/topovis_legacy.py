#!/usr/bin/env python3

"""
####################################################################################################################################################################################
##############################################################  TOPOVIS.PY #########################################################################################################
                                                    TOkamak POloidal cross section VISualisation


    Purpose:
        This code is designed to plot cross sections of the tokamak at certain toroidal angles. It works for linear and nonlinear simulations. 
        Implemented geometries are circular, global CHEASE and s-alpha. 
        For nonlinear simulations, the code can plot the zonal as well as the non-zonal potential using two different interpolation methods: B-Spline interpolation
        or interpolation with a Fast-Fourier-Transformation. The user will be asked which potential (zonal or non-zonal) to plot and which interpolation method to use.
    Input: 
        - gkwdata.h5 file
        - toroidal angle: phi_val
        - optional: timestep of potential, poten_timestep
    Output:
        - pdf file: poloidal cross section of the tokamak at phi_val
        - h5 file: data for plot of poloidal slice: (R, Z, potential)
    Terminal command: 
        python topovis.py <filepath>/gkwdata.h5 <phi_val>
        or:
        python topovis.py <filepath>/gkwdata.h5 <phi_val> <poten_timestep>
    Author:
        Sofia Samaniego
        Universitaet Bayreuth
        01.08.2024

"""


#################################################################################################################################################################################
############################################################## PACKAGES #########################################################################################################

import h5py 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from matplotlib.tri import Triangulation
from scipy.integrate import trapezoid   # As of scipy v1.12.0 scipy.integrate.trapz has been deprecated in favor of trapezoid
import argparse
import os
pi = np.pi

#################################################################################################################################################################################
############################################################## FUNCTIONS ########################################################################################################

def make_zeta_grid(n_spacing, mphi): 
    """
    Make zeta grid for whole torus (not just 1/n_spacing of torus). This will only be used in the nonlinear case. 

    Args:
        n_spacing (int): Smallest mode number of toroidal modes that are resolved in the simulation.
        mphi (int): Number of grid points in Zeta direction for simulated part of torus. 
    Returns:
        ndarray: The zeta grid. Shape is (mphi*n_spacing,).
    """
    L_zeta = 1/n_spacing
    delta_zeta = L_zeta / mphi
    zeta = np.array([j*delta_zeta for j in range((n_spacing*mphi))])
    return zeta

def make_zetashift(geomtype, s_grid, gmap, q_val, phi_val, R, eps_idx, N_s, signB, signJ):
    """
    Generate the grid for zeta which preserves phi = const. 

    Args:
        geomtype (str): The type of geometry. Can be 'circ', 's-alpha', or 'chease_global'.
        s_grid (ndarray): The grid of s values.
        gmap (ndarray): All g values. Shape is (N_s,).
        phi_val (float): The value of phi.
        R (ndarray): The grid of R values. Shape is (NX, N_s).
        eps_idx (int): The index of epsilon.
        N_s (int): The number of s values.
        signB (float): The sign value of B.
        signJ (float): The sign value of J.

    Returns:
        ndarray: The zetashift grid which is phi-preserving. It will be mapped back to 0 <= zeta <= 1 using '%1'.

    Note:
        - For 'circ' geometry, the zeta shift is calculated as (-phi_val / (2 * pi) + g) % 1.
        - For 's-alpha' geometry, the zeta shift is calculated as (signB*signJ/(2*pi)*(2*pi*abs(q_val)*s-phi_val)) % 1.
        - For 'chease_global' geometry, the zeta shift is calculated using numerical integration (trapezoidal rule). 
          zeta(s) = -phi/2pi + gmap * integral_0^s (1/(R**2)). Also: R is not the unitless R as for circ geom, but R*R_REF. 
    """
    import time
    if geomtype == 'circ':
        zetashift = np.array([(- phi_val / (2 * pi) + g) % 1 for g in gmap])
        return zetashift 
    
    elif geomtype == 's-alpha':
        zetashift = np.array([(signB*signJ/(2*pi)*(2*pi*abs(q_val)*s-phi_val)) % 1 for s in s_grid ])
        return zetashift   

    elif geomtype == 'chease_global':
        Rsliced = R[eps_idx, :]

        # even amount of s-values
        if len(s_grid) % 2 == 0:
        # The integration starts at 0. For even number of s-values, there is no s=0. R(s=0) is interpolated (B-Spline interpolation with k=3). 
        # Supporting point (R(s=0), s=0) has to be added to the other supporting points to perform integration.
            spl = splrep(s_grid, Rsliced, k = 3, s=0) 
            s_0 = np.array([0]) # s = 0
            R_0 = np.array(splev(s_0, spl)) # R(s=0)

            zetashift = np.zeros((N_s))
            n_neg = len(s_grid[s_grid<0]) # number of s_grid-points smaller than 0

            # s < 0
            for s in range(n_neg):  # 0...n_neg-1                       
                zetashift[s] = - phi_val/(2*pi) + g[eps_idx, s] * (-trapezoid(              # -trapezoid as integration goes from 0 to s, but here it is integrated from s to 0 as s is negative
                    1/((np.concatenate((Rsliced[s_grid<0][s:], R_0)) * R_REF) ** 2),     # (1/(R**2)) as integrand
                    np.concatenate((s_grid[s_grid<0][s:], s_0))                         # s <= 0 are sample points
                    )
                )          

            # s >= 0
            for s in range(n_neg, N_s): # n_neg...N_s-1
                zetashift[s] = - phi_val/(2*pi) + g[eps_idx, s] * trapezoid(                # trapezoid as integration goes from 0 to s and s is positive
                    1/((np.concatenate((R_0, Rsliced[n_neg:s+1])) * R_REF) ** 2),        # (1/(R**2)) as integrand
                    np.concatenate((s_0, s_grid[n_neg:s+1]))                            # s >= 0 are sample points
                )
        else:
        # For odd amount of s values: s=0 exists, it does not have to be appendend as before.
            zetashift = np.zeros((N_s))
            n_neg = len(s_grid[s_grid<0]) # number of s_grid-points smaller than 0

            # s <= 0
            for s in range(n_neg+1):  # 0...n_neg                       
                zetashift[s] = - phi_val/(2*pi) + g[eps_idx, s] * (-trapezoid(   
                    1/((Rsliced[s_grid<=0][s:] * R_REF) ** 2), 
                    s_grid[s_grid<=0][s:]   
                ))

            # s >= 0
            for s in range(n_neg, N_s): # n_neg...N_s-1
                zetashift[s] = - phi_val/(2*pi) + g[eps_idx, s] * trapezoid(
                    1/(( Rsliced[n_neg:s+1] * R_REF) ** 2), 
                    s_grid[n_neg:s+1]              
                )
            # zeta for s=0 is calculated two times. It does not matter as it is overwritten in the second loop.
            # s=0 has to be part of both ranges as we need that supporting point for integration.
        
        return zetashift % 1 # mapping zeta to [0, 1]

def make_Bspline_interpolation(x, y, num, x_ev):
    """
    Perform B-Spline interpolation to calculate the values of y at specified x values.

    Args:
        x (ndarray): The x-coordinates of the interpolation points.
        y (ndarray): The y-coordinates of the interpolation points. Shape should be (num, len(x)).
        num (int): The number of (x,y) pairs.
        x_ev (ndarray): The x-coordinates at which the interpolated y_ev values are to be calculated.

    Returns:
        ndarray: The interpolated y_ev values at the specified x_ev coordinates. Shape is (num,).
    """

    y_ev = np.zeros(num) 
    spl = [splrep(x, y[i, :], k=3) for i in range(num)]             # calculate spline coefficients for each set of y values
    y_ev = np.array([splev(x_ev[i], spl[i]) for i in range(num)])   # evaluate interpolated values at x_ev points using spline interpolation
    return y_ev

def whole_potential(n_spacing, potential_data, eps_idx): 
    """
    Returns the potential on the whole torus for a specific epsilon value as potential_data is only for (1/nspacing)-part of the torus.

    Args:
        n_spacing (int): The number of torus sections.
        potential_data (ndarray): The 3D potential data for in GKW simulated part of torus.
        eps_idx (int): The index of the epsilon value.

    Returns:
        ndarray: The potential on the whole torus for the given epsilon value.
    """

    slice = potential_data[:, eps_idx, :] 
    whole = slice.copy()
    for _ in range(int(n_spacing-1)):
        whole = np.concatenate((whole, slice), axis=1)
    return whole 

def reshape_parallel_dat_multi(file, nmod, NX, N_s):
	"""
	Reshapes the parallel data from the gkwdata.h5 file based on the given parameters.
    It only takes the last from GKW saved parallel.dat. 

	Args:
	    file (h5py.Group): The gkwdata.h5 file containing the parallel data.
	    nmod (int): The number of modes.
	    NX (int): The amount of grid points in the radial direction (eps).
	    N_s (int): The amount of grid points in the s direction.

	Returns:
	    ndarray: The reshaped parallel data.
	"""
	nsp = np.array(file['input/grid/number_of_species'])        # number of species
	ncol = 15                                                   # number of columns in parallel.dat for first dimension
	par = file["diagnostic/diagnos_mode_struct/parallel"][()]		
	n_parallel = int(par.shape[1]/(nsp*nmod*NX*N_s)) 	        # get number of parallel.dat's in current hdf5-file
	end_idx = par.shape[1]                                      # indizes of last parallel.dat (from start_idx to end_idx)
	start_idx = int((n_parallel-1)*(nsp*nmod*NX*N_s))
	outData = np.reshape(par[:, start_idx:end_idx], (int(ncol), int(nsp), int(nmod), int(NX), int(N_s))) # only takes last parallel.dat
	return outData

def fouriercoeff_imag(file, imode, nmod, NX, N_s):
    """
    Retrieves the imaginary part of the Fourier coefficients from the parallel data in the given file.
    Here, it is used to retrieve PHI_IMAG.

    Args:
        file (h5py.File): The HDF5 file containing the parallel data.
        imode (int): The mode index.
        nmod (int): The number of modes.
        NX (int): The number of grid points in the radial direction.
        N_s (int): The number of grid points in the s direction.

    Returns:
        ndarray: The imaginary part of the Fourier coefficients.  

    Note:
        This function can be used for different quantities in parallel.dat:
        SGRID = 0
        PHI_REAL = 1
        PHI_IMAG = 2
        APAR_REAL = 3
        APAR_IMAG = 4
        DENS_REAL = 5
        DENS_IMAG = 6
        UPAR_REAL = 11
        UPAR_IMAG = 12
        ENE_PAR_REAL = 7
        ENE_PAR_IMAG = 8
        ENE_PERP_REAL = 9
        ENE_PERP_IMAG = 10
        Then, imag_idx has to be adjusted accordingly. 

        Dimensions of reshaped parallel.dat:
        COL_DIM = 0
        NSP_DIM = 1
        NMOD_DIM = 2
        NX_DIM = 3
        NS_DIM = 4

    """
    PHI_IMAG = 2        # identify column
    isp = 0             # just one species
    parallel = reshape_parallel_dat_multi(file, nmod, NX, N_s)
    imag_idx = PHI_IMAG
    data_imag = parallel[imag_idx, isp, imode, :, :]
    return data_imag

def fouriercoeff_real(file, imode, nmod, NX, N_s):
    """
    Retrieves the real part of the Fourier coefficients from the parallel data in the given file.
    Here, it is used to retrieve PHI_REAL. 

    Args:
        file (h5py.File): The HDF5 file containing the parallel data.
        imode (int): The mode index.
        nmod (int): The number of modes.
        NX (int): The number of grid points in the radial direction.
        N_s (int): The number of grid points in the s direction.

    Returns:
        ndarray: The imaginary part of the Fourier coefficients. 

    Note:
        This function can be used for different quantities in parallel.dat. See fouriercoeff_imag. 

    """
    PHI_REAL = 1    # identify column
    isp = 0         # just one species
    parallel = reshape_parallel_dat_multi(file, nmod, NX, N_s) 
    real_idx = PHI_REAL
    data_real = parallel[real_idx, isp, imode, :, :]
    return data_real

#################################################################################################################################################################################
############################################################## ARGPARSE #########################################################################################################
"""
Terminal Input: python topovis.py hdf5-filename phi_val zonal_potential poten_timestep(optional)
For example: python topovis.py ../gkwdata.h5 3.1 True 4
For help: python topovis.py --help
Timestep input has to be >=1. Counting starts at 1.
"""
    
parser = argparse.ArgumentParser(description='Visualisation of a poloidal cross section for given toroidal angle phi and time step for potential data. Zonal or non-zonal potential is displayed.')
parser.add_argument('hdf5_filename', type=str, help='File path to hdf5 File.')
parser.add_argument('phi_val', type=float, help='Value for toroidal angle phi.')
parser.add_argument('poten_timestep', type=int, nargs='?', help='Time step for potential. If it is out of range, the last time step is used.')
args = parser.parse_args()

# arguments and correct filetype 
hdf5_filename = args.hdf5_filename
phi_val = args.phi_val
filetype = hdf5_filename.split(".")[-1]
if filetype != 'h5':
    print('Input file is not an HDF5 file. Code will exit.')
    os._exit(0)

# hdf5 File
hdf5_file = h5py.File(hdf5_filename, "r")

# linear or nonlinear simulation
non_linear_array = hdf5_file["input/control/non_linear"][()]
non_linear_string = non_linear_array[0].decode('utf-8')

# 3D potential data is only needed for nonlinear data
if non_linear_string == 'T':

    # default potential data set for nonlinear simulations
    poten_keys = [key for key in hdf5_file["diagnostic/diagnos_fields"].keys() if key.startswith("Poten")]
    dataset_name = poten_keys[-1]
    potential_data_original = hdf5_file[f"diagnostic/diagnos_fields/{dataset_name}"] 
    potential_data = np.array(potential_data_original)

    # use correct potential data set if poten_timestep is given and in range
    if args.poten_timestep is not None:
        poten_timestep = args.poten_timestep
        poten_timestep -= 1 # index of list starts with 0
        if 0 <= poten_timestep <= len(poten_keys): 
            dataset_name = poten_keys[poten_timestep]
            potential_data_original = hdf5_file[f"diagnostic/diagnos_fields/{dataset_name}"]
            potential_data = np.array(potential_data_original)
    
#################################################################################################################################################################################
############################################################## MAIN #############################################################################################################

# -----------------------------------------------------------------------------------------------------------------------------------------
# values from file
R = hdf5_file["geom/R"][()] 
R_REF = float(hdf5_file["geom/Rref"][()][0])
Z = hdf5_file["geom/Z"][()] 
s_grid = hdf5_file["geom/s_grid"][()] 
eps = hdf5_file["geom/eps"][()] 
q = hdf5_file["geom/q"][()] 
n_spacing = int(hdf5_file["input/mode/n_spacing"][()][0])
nmod = int(np.array(hdf5_file['input/grid/nmod'])[()][0])
mphi = int(hdf5_file["diagnostic/diagnos_grid/mphi"][()][0])  # number of grid points in Zeta direction
g =  hdf5_file["geom/gmap"][()]                               
rhostar = hdf5_file["input/spcgeneral/rhostar"][()][0]        # normed Larmor radius
N_s = int(hdf5_file["input/grid/n_s_grid"][()][0])            # number of grid points in s direction
NX = int(hdf5_file["input/grid/n_x_grid"][()][0])             # number of grid points in psi (radial) direction
signB = hdf5_file["input/geom/signB"][()][0]
signJ = hdf5_file["input/geom/signJ"][()][0]

R = np.reshape(R, (NX, N_s))
Z = np.reshape(Z, (NX, N_s))
g = np.reshape(g, (NX, N_s))

# -----------------------------------------------------------------------------------------------------------------------------------------

# geom type and exit for geometries other than circ, chease_global or s-alpha
geom_type = hdf5_file["input/geom/geom_type"][()][0].decode('utf-8')
if geom_type != 'circ' and geom_type != 'chease_global' and geom_type != 's-alpha':
    print('Other geometrys other than circular, global chease or a-alpha geometry are not implemented yet. Code will exit.')
    os._exit(0)

# mapping back phi to 0 <= phi <= 2pi
phi_val = phi_val % (2 * pi)

# user interaction for zonal/non-zonal pot, FFT/Splines
# It is recommended to use Splines rather than FFT, as it performed better in the benchmarks.
if non_linear_string == 'T':
    # Zonal or non-zonal potential
    zonal_potential = input('Zonal or non-zonal potential? Enter 1 for zonal, 0 for non-zonal: ')
    if zonal_potential in ('1', 'true', 't', 'yes', 'y', '1', 'True'):
        zonal_potential = True
    elif zonal_potential in ('0', 'false', 'f', 'no', 'n', '0', 'False'):
        zonal_potential = False
        pot_mean = np.mean(np.mean(potential_data, axis=2), axis=0)  # shape (NX,)
        potential_data = potential_data - pot_mean[None, :, None]    # non-zonal potential
    else:
        print('Boolean value expected. Code will exit.')
        os._exit(0)
    print('Zonal potential: ', zonal_potential)
    int_method = input('Interpolation via B-Splines or FFT? Enter 1 for B-Splines, 0 for FFT: ')

# initialize final data output array for R, Z and potential for every s and eps
data = np.zeros((N_s*NX, 3)) 

# linear simulation with nmod = 1: Fourier coefficients from parallel.dat
if non_linear_string == 'F':
    coeff_imag_entire = np.zeros((nmod, NX, N_s))
    coeff_real_entire = np.zeros((nmod, NX, N_s))
    coeff_imag_entire = fouriercoeff_imag(hdf5_file, 0, nmod, NX, N_s) # imod = 0
    coeff_real_entire = fouriercoeff_real(hdf5_file, 0, nmod, NX, N_s) 

# Grid for zeta
zeta_grid = make_zeta_grid(n_spacing, mphi) 

# For nonlinear case with FFT, create array for k_zeta 
if non_linear_string == 'T' and int_method == '0':
    k_pos = np.array([2*pi*n_spacing*mod for mod in range(1, nmod)])
    del_k = 2*pi*n_spacing # space between two k values

    # extend k 
    k_pos_neg = np.concatenate((k_pos, -k_pos))     # append negative k vectors
    k_extended = np.concatenate((k_pos_neg, [0]))   # adding nmod = 0 
    k_final = np.sort(k_extended) 

    # zero idx is at half the max index for even nmod, for odd nmod it is in the middle of the array
    # zero idx is equal to the amount of values for negative k, same amount of positive and negative k (symmetry)
    zero_idx = mphi // 2 
    
    # extend k more (because of zero Padding)
    # left side
    while np.sum(k_final < 0) < zero_idx:
        min_negative_k = np.min(k_final[k_final < 0])   # smallest k value
        k_final = np.concatenate((k_final, [min_negative_k - del_k]))
    k_final = np.sort(k_final)

    # right side
    while np.sum(k_final > 0) < zero_idx:
        max_positive_k = np.max(k_final[k_final > 0])   # biggest k value
        k_final = np.concatenate((k_final, [max_positive_k + del_k]))
    k_final = np.sort(k_final)

############################################################## CALCULATE POL SLICE ###############################################################################################

# two cases: linear with nmod = 1 and nonlinear (options for the latter: FFT and B-Splines)
if non_linear_string == 'F' and nmod ==1 :
    print('Linear simulation')
    for eps_idx in range(NX):
        # zeta for phi = const.
        zetashift = make_zetashift(geom_type, s_grid, g[eps_idx,:], q[eps_idx], phi_val, R, eps_idx, N_s, signB, signJ) 

        # Fourier coefficients
        coeff_imag = coeff_imag_entire[eps_idx, :]
        coeff_real = coeff_real_entire[eps_idx, :]  # \hat(f) = coeff_real + i coeff_imag
        coeff = coeff_real + 1j * coeff_imag        # Fourier coefficients, shape (N_s,)
        
        # wave vector
        # as nmod = 1: there is only one k
        k = 2*pi*1*n_spacing                        

        # potential for this eps
        curr_pot = np.zeros((N_s)) 

        # Calculate the exponential terms
        exp_pos = np.exp(1j * k * zetashift)  # shape (N_s,)
        exp_neg = np.exp(-1j * k * zetashift) # shape (N_s,)

        # Calculate interpolated potential curr_pot with vector operations
        values = coeff * exp_pos + np.conjugate(coeff) * exp_neg
        curr_pot = values.real

        # save data
        start_idx = eps_idx * N_s 
        end_idx = start_idx + N_s
        data[start_idx:end_idx, 0] = R[eps_idx, :]
        data[start_idx:end_idx, 1] = Z[eps_idx, :]
        data[start_idx:end_idx, 2] = curr_pot

elif non_linear_string == 'T':
    print('Nonlinear simulation')
# -----------------------------------------------------------------------------------------------------------
    if int_method == '1': # nonlinear case with B-Splines
        print('Interpolation method: B-Splines')
        for eps_idx in range(NX):
            zetashift = make_zetashift(geom_type, s_grid, g[eps_idx, :], q[eps_idx], phi_val, R, eps_idx, N_s, signB, signJ) # zeta for phi = const.
            potential = whole_potential(n_spacing, potential_data, eps_idx)
            curr_pot = make_Bspline_interpolation(zeta_grid, potential, N_s, zetashift) # interpolated potential for this eps, shape (N_s,)

            # save data
            start_idx = eps_idx * N_s 
            end_idx = start_idx + N_s
            data[start_idx:end_idx, 0] = R[eps_idx, :]
            data[start_idx:end_idx, 1] = Z[eps_idx, :]
            data[start_idx:end_idx, 2] = curr_pot

# -----------------------------------------------------------------------------------------------------------
    elif int_method == '0': # nonlinear case with Fourier transformation
        print('Interpolation method: FFT')
        for eps_idx in range(NX):
            zetashift = make_zetashift(geom_type, s_grid, g[eps_idx, :], q[eps_idx], phi_val, R, eps_idx, N_s, signB, signJ) # zeta for phi = const.
            curr_pot = np.zeros((N_s)) # interpolated potential for this eps, shape (N_s,)
            potential = potential_data[:, eps_idx, :]

            # Fourier transformation and shift along zeta
            fft_results = np.fft.fftshift(np.fft.fft(potential, axis=1), axes=1)  # s is axis 0, k is axis 1

            # even nmod: one more value for negative k than for positive k -> make it symmetrical
            # append conjugate of value for smallest k at the end (corresponding to biggest k)
            if fft_results.shape[1] % 2 == 0:
                fft_results = np.append(fft_results, fft_results[:, 0:1].conjugate(), axis=1)

            # for all s and k: calculating the exp terms with vectorization
            exp_terms_pos = np.exp(1j * k_final[None, :] * zetashift[:, None]) # shape (s, k)
            exp_terms_neg = np.exp(-1j * k_final[None, :] * zetashift[:, None])

            # calculate curr_pot with vector operation
            curr_pot[:] = np.real(fft_results[:, zero_idx] * exp_terms_pos[:, zero_idx]) # k = 0
            k_idxs = np.arange(zero_idx + 1, len(k_final)) # indizes of k for k > 0
            curr_pot += np.sum(np.real(fft_results[:, k_idxs] * exp_terms_pos[:, k_idxs] + fft_results[:, k_idxs].conjugate() * exp_terms_neg[:, k_idxs]), axis=1) # sum over all k > 0: axis = 1
            curr_pot /= len(k_final) # transforming it back

            # save data
            start_idx = eps_idx * N_s 
            end_idx = start_idx + N_s
            data[start_idx:end_idx, 0] = R[eps_idx, :]
            data[start_idx:end_idx, 1] = Z[eps_idx, :]
            data[start_idx:end_idx, 2] = curr_pot

    else:
        print('0 or 1 expected for interpolation method. Code will exit.')
        os._exit(0)
else:
    print('Simulations other than linear with nmod=1 or nonlinear cannot be dealt with. Code will exit.')
    os._exit(0)

#################################################################################################################################################################################
############################################################## PLOT #############################################################################################################

phi_val_str = str(round(phi_val, 4)).replace('.', '-')
# plot layout 
#plt.rc('text', usetex=True)
#plt.rc('font', family='lmodern') 

# make heat map
cbar_limit = max(np.abs(data[:, 2].min()), np.abs(data[:, 2].max())) 
num_levels = 200
triangles = Triangulation(data[:, 0], data[:, 1]).triangles
plt.tricontourf(data[:, 0], data[:, 1], triangles, data[:, 2], cmap='seismic', vmin=-cbar_limit, vmax=cbar_limit, levels=num_levels) # diverging color map

# minimal eps -> there must be hole in the middle
plt.plot(np.append(R[0, :], R[0, 0]), np.append(Z[0, :], Z[0, 0]), color='grey', alpha=0.3, linewidth=1)
plt.fill(R[0, :], Z[0, :], color='white')

# plot outline
plt.plot(np.append(R[-1, :], R[-1, 0]), np.append(Z[-1, :], Z[-1, 0]), color='grey', alpha=0.3, linewidth=1)

# figure layout stuff
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20) 
cbar.set_label(label=r'$\phi(R, Z)$', fontsize=26)
# plt.xlabel('R', fontsize=26)
# plt.ylabel('Z', fontsize=26)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
plt.grid(False)
plt.axis('equal')
# plt.title(r'Poloidal Cross Section for $\varphi={}~^\circ$'.format(round(phi_val, 4)), fontsize=16)

if non_linear_string == 'T':
    plt.savefig(f'polslice-{phi_val_str}-{zonal_potential}-{dataset_name}.pdf', dpi=300, bbox_inches='tight')
else:
    plt.savefig(f'polslice-{phi_val_str}.pdf', dpi=300, bbox_inches='tight')

#################################################################################################################################################################################
############################################################## SAVE DATA ########################################################################################################

if non_linear_string == 'T':
    f = h5py.File(f"topovisdata-{phi_val_str}-{zonal_potential}-{dataset_name}.h5", "w")
else:
    f = h5py.File(f"topovisdata-{phi_val_str}.h5", "w")

lng = len(data) # length of data sets, N_s*NX

dset_R = f.create_dataset("R", (lng,), 'f')
dset_R[()] = data[:, 0]

dset_Z = f.create_dataset("Z", (lng,), 'f')
dset_Z[()] = data[:, 1]

dset_Pot = f.create_dataset("Poten", (lng,), 'f')
dset_Pot[()] = data[:, 2]

# add for debugging
f.create_dataset("s", data=s_grid)
f.create_dataset("x", data=eps)

f.close()
