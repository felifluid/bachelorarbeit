#import "@preview/subpar:0.2.1"

= Improving the ToPoVis Code <chap:improvement>

// TODO: passt die position hier?
#include "../../figs/triangulation_artifacts/sparse/artefacts.typ"; <fig:numerical_artifacts>

== Dealing with Numerical Artifacts
=== The Cause of Numerical Artifacts
In some cases, the original version of ToPoVis creates numerical artifacts in the contour plots as can be seen in @fig:numerical_artifacts. 
The artifacts are particularly severe at the edges of plots in CHEASE geometry. 
This made it harder to study small scale turbulences in these areas.
Fixing these artifacts is the original motivation underlying this thesis. 

As mentioned in @sec:topovis:plotting, ToPoVis uses the function `tricontourf` from the package `matplotlib.pyplot` to create the heatmap plots.
As the name suggests it uses an unstructured triangle mesh to draw contour regions.
If no triangle grid is supplied, it will generate one implicitly using a delaunay algorithm @matplotlib2025tricontourf. 
This generation of smooth surfaces using triangulation is commonly used as a linear interpolation of (un-)structured three-dimensional data. 
The method `tricontourf` uses the triangle mesh to interpolate between the discrete unstructured points.
As such, the choice of triangulation has a big impact on the resulting interpolation. 
As discussed in @sec:triang, delaunay triangulation usually yields the best results, however, it faces issues with heavily non-uniform grids.
The detrimental effect of this in ToPoVis can be observed, when taking a look at the underlying grid structure and resulting delaunay triangulation in @fig:artifacts:sparse:delaunay.

#include "../../figs/triangulation_artifacts/sparse/delaunay/fig.typ"

@fig:artifacts:sparse:delaunay and @fig:artifacts:sparse:regular[] show a subsection of simulation data in CHEASE geometry outputted by ToPoVis. 
The section stands out with a really low density of the $s$-grid and a high density in the radial direction and can therefore be classified as a heavily non-uniform grid.
As explained in the previous @sec:triang,
non-uniform distributions can lead to so called "fat" triangles in areas of low density, as well as many strongly acute triangles surrounding them.
This is exactly what can be observed in @fig:artifacts:sparse:delaunay:grid.
Note that the axes aren't scaled equally to help visualize this effect. 
Furthermore only every fourth point in the  $psi$-direction is used to make the triangles more distinguishable. 
This has little to no influence on the delaunay triangulation in this specific section.

=== An alternative Triangulation
#include "../../figs/triangulation_artifacts/sparse/regular/fig.typ"

A possible solution to this is presented in @fig:artifacts:sparse:regular up top.
Instead of relying on delaunay triangulation, a custom _regular_ triangle grid is used.
To achieve this, triangulation must be done in the equidistant Hamada coordinates instead.
The poloidal coordinates $R(psi,s)$ and $Z(psi,s)$ are exported by GKW as flattened arrays.
As both coordinates are defined through the discrete hamada grid, they can also written as $R_(i j)$ and $Z_(i j)$, with $i,j$ being the indices of $psi$ and $s$ respectively.
Therefore, every distinct poloidal point ${R_(i j), Z_(i j)}$ can be represented in the regular hamada coordinates $(psi_i,s_j)$ instead.

// TODO: Überleitung?

Consider an equally spaced and rectangular grid of points

$ P_(i j) $

defined by the two indices $i,j$

$ 0<=i<=n -1, #h(1em) 0 <= j <= m -1 $

where both $n>=2$ and $m>=2$.

Then we want to create a triangle grid, in such a way as can be seen in @fig:triangulation:regular_grid below.

#include "../../figs/triangulation/regular_grid/fig.typ"

The list of triangles can be described mathematically as

$ T = sum_(i=0)^(n-2) sum_(j=0)^(m-2) overbracket([(i,j), (i+1, j), (i, j+1)], #sym.triangle.tl) + overbracket([(i+1, j), (i+1, j+1), (i, j+1)], triangle.stroked.br) $ <eq:triangles_tuple>

The choice of the diagonal direction can be chosen arbitrarily, i.e. either _right_ (as in @fig:triangulation:regular_grid) or _left_.

However, plot functions like `tricontourf` do not support tuple representation $(i,j)$ of points as in @eq:triangles_tuple.
Instead points have to be addressed by a singular index $k$, which represents their position in a _flattened_ array $P_k$, meaning that

$ k_(i j) = i + j dot n $ <eq:triangle:flattened_index>

Therefore, @eq:triangles_tuple can be rewritten to

$ T = sum_(i=0)^(n-2) sum_(j=0)^(m-2) overbracket([k_(i j), k_(i+1, j), k_(i, j+1)], #sym.triangle.tl) + overbracket([k_(i+1, j), k_(i+1, j+1), k_(i, j+1)], triangle.stroked.br) $ <eq:triangles:flattened>

In the context of poloidal periodicity another row of triangles has to be added.
This is equivalent to adding the following term to the above formula

$ sum_(i=0)^(m-2) overbracket([k_(i, m-1), k_(i+1,m-1), k_(i, 0)],triangle.tl) + overbracket([k_(i+1, m-1), k_(i+1,0), k_(i, 0)], triangle.br) $

again assuming index $j$ describes the poloidal $s$ coordinate.
Doing so adds triangles between the first $j=0$ and last $j=m-1$ values for all radial indices $i$.

In code this is done by the new method `make_regular_triangles` via list comprehentions.

=== Limits of Triangulation <sec:triang:limits>

While regular triangulation provides better results in some areas, it can lead to unfavorable triangles in different ones. 
The CHEASE geometry is _sheared_ poloidal in $theta$ along the radial coordinate $psi$. 
However, the shearing is asymmetrical in the poloidal coordinate $s$.
Around $s=0$ the shearing is minimal, so lines with constant s are nearly straight radially.
The shearing reaches its maximum at approximately $s=±0.5$. 
In this area the constant-s-lines curve counter-clockwise.
Because of this several $s="const"$ lines span poloidal parallel to each other.
This leads to an unstructured, high density grid, that is partially uniform, which is favorable for delaunay triangulation.

#include "../../figs/triangulation_artifacts/sheared/fig.typ"

@fig:sheared_triang shows four plots. 
The two on the top use delaunay triangulation, the bottom two make use of the `make_regular_triangles` method, each with a contour plot and a triangulation plot.  
Even though the values of the discrete grid points stay the same, the resulting contour plot changes heavily between triangulation methods. While the contour plot using delaunay triangulation is characterized with big horizontal stripes, the regular contour plot consists of many thin vertical stripes. This is directly correlated with the shape and orientation of the triangulations. 

As interpolation is done along the edges of the triangles, the resulting contour plot might resemble the triangulation in its structure.
Both triangulations operate on _nearest neighbor_ approaches. 
However, while the delaunay algorithm leads to points being connected to near neighbors in _poloidal_ coordinates, the regular triangulation defines the distance of two points in _hamada_ coordinates. 
Two points can be close to each other in hamada coordinates, while being really distant poloidally. 
That is why the regular triangulation can be considered as an interpolation in hamada coordinates, and the delaunay triangulation as a interpolation in poloidal coordinates. 
There cannot be made a clear decision in favor of neither of the two approaches, without comparing the results with simulations with higher resolutions. 
Findings in @sec:topovis:interpolation show, that interpolations in hamada coordinates are coinciding better with the high resolution simulations.

A more and refined way to improve triangulations is to avoid the causes of unfavorable triangulations in the first place. 
It can be achieved by interpolating the grid by other means _before_ triangulation is done, which will be discussed next.

// the best method would probably be to use delaunay refinement in combination with interpolation -> outlook

// of course simulations with higher density in s would help. but those are really heavy to compute. for convergence of the simulation only relatively few s grid points are needed.

=== Refining the Grid through Interpolation <sec:topovis:interpolation>

Similarly to the triangulation, the interpolation of the grid can be done in both poloidal coordinates and hamada coordinates. 
For now poloidal interpolation is only implemented for linear simulations, as interpolation in hamada coordinates currently yields more promising results.
In this section, it will be explained in detail, how interpolation is achieved both for linear and non-linear simulations.
For validation and benchmarking the interpolators are tested on simulations with a low-density s-grid in both circular and CHEASE geometry.
The results will then be compared to simulation data with a s-grid resolution. 

==== Linear Simulations
As explained in @sec:topovis:linear, the potential on a poloidal slice in the linear case is calculated by first calculating $zeta$-shift.
The potential is then calculated using the $zeta$-shift and the complex fourier coefficients using the formula

$ Phi(hat(f), zeta, k) = hat(f) dot e^(i k zeta) + hat(f)^* dot e^(-i k zeta) $

This leaves open two different strategies for interpolating the potential:

#set enum(numbering: "A)", indent: 1em)
+ Calculate the potential on the sparse grid and interpolate it.
+ First interpolate $zeta$\-shift and $hat(f)$ and then calculate the potential on the fine grid.

While option A) is the simpler approach, it also generates worse results. 
This is presumably due to the potential being a real number.
So before interpolation the imaginary part of the complex fourier coefficients gets discarded.
Whereas the imaginary part is being kept and interpolated in option B).
Additionally, $hat(f)$ and $zeta$\-shift each represent smoother surfaces than the potential, which makes interpolation less prone to numerical errors.
This makes option B) the preferred approach, while option A) will not be discussed or used further in this thesis.

There is still the choice of the coordinate system to interpolate in. 
Both poloidal and Hamadian interpolation will be discussed in the next sections.

===== Hamadian Interpolation <sec:interpolation:hamada>
Interpolation in hamada coordinates is done using `RegularGridInterpolator`, which functionality is discussed in @sec:background:rgi. 
Both $zeta$-shift and the fourier coefficients $hat(f)$ are defined as 2d-arrays as functions of $psi$ and $s$.
Therefore interpolation is a simple act of first initializing the interpolator with the sparse grid and data and then calling it to evaluate on the fine grid.

```py
    rgi = RegularGridInterpolator((x, s), data, **kwargs)
    data_fine = rgi(xs_points_fine)
```
The main restriction with this method is, that the RGI can only _interpolate_. 
This wouldn't be an issue, if the s-grid is defined as $s=[-0.5, 0.5]$. 
But due to specification of the discrete grid (see @sec:discrete_grid), it is defined from $s_0=-0.5 + (Delta s) / 2$ to $s_(-1) = 0.5 - (Delta s)/2$. 
This leads to a gap of exactly $Delta s$ between $s_0$ and $s_(-1)$.
In order to make interpolation work in this gap, the grid has to be extended outside the bounds of $-0.5 < s < 0.5$. 

In itself $s$ is perfectly periodic, meaning $s=s±1$. 
However, the RGI depends on a strictly ascending or descending grid. 
So instead of wrapping the s-grid periodically, it needs to be extended out of bounds with regular spacing. 
In code this is achieved by the method `extend_regular_array(a: arr, n: int)`, which first checks if the given array is equally spaced, and then extends it by `n` in both directions. 
The extended s-grid is defined from $s_0=-0.5-n dot (Delta s)/2$ to $s_(-1) = 0.5 + n dot (Delta s)/2$. 
It is only used to define the _virtual_ position of the periodically extended data points for the RGI.

For $zeta$-shift and $hat(f)$ double periodic boundary conditions apply (see @sec:background:hamada:periodicity)

$ 
    &zeta(s) = zeta(s±1) ∓ q \
    &hat(f)(s) = hat(f)(s±1) dot e^(±i k q)
$

// reference exact equations??

In code this is done by first extending the grid periodically, and then applying the boundary condition to the extended parts respectively, e.g. in the case of $zeta$-shift

```py
    # extend grid periodically in s
    zeta_s = extend_periodically(zeta_s, OVERLAP, 1)
    # apply boundary condition for zeta[s < -0.5]
    zeta_s[:, :OVERLAP] -= dat.q[:, None]               
    # apply boundary condition for zeta[s > 0.5]
    zeta_s[:, -OVERLAP:] += dat.q[:, None]
```

After applying these parallel periodic boundary conditions, both $zeta$-shift and $hat(f)$ are continuous across $s=±0.5$.
This makes interpolation possible across the extended s-grid, which enables interpolating in the gap.
Note that for interpolation of order 3 or higher, the grid has to be extended by more than one point. 
In code this is controlled by the constant `OVERLAP`.

===== Poloidal Interpolation
Interpolation in poloidal coordinates works quiet similar as interpolation is hamada coordinates.
But instead of initializing the interpolator with the grid in hamada coordinates, the data points are paired with the corresponding scattered R-Z-points.

The challenge of interpolating in poloidal coordinates lies in the scattered structure of the grid. // TODO: ist das gutes Englisch?
This applies especially to data in CHEASE geometry, which shows multiple different non-uniform characteristics.
Intuitively interpolation in poloidal coordinates seems like a good approach, as the coordinates describe the real world euclidean space.
If two points are really close to each other in euclidean space, one can expect similar measurements at these points. // ??: gibts dafür ein Fachbegriff?
As the electric potential is smooth and continuous inside the tokamak, this property also applies to it. // TODO: dont like this sentence
Interpolating in hamada coordinates does not take the euclidean distance into account. 
Due to the non-linear transformation between the two coordinate systems, two points can be close to each other in hamadian space, but have a large distance in euclidean space.

Many interpolation methods like multivariate splines, finite element or Clough-Tocher rely on mesh generation through triangulation.
This makes them unsuitable for this purpose, as triangulation algorithms fail to produce reliable results on non-uniform grids. 
Therefore meshfree methods like the `RBFInterpolator` are can be used to create more uniform grids and be able to generate triangulations without numerical artifacts.
The functionality of the RBFI is discussed in detail in @sec:background:rbfi.

Using the RBFI, interpolating in poloidal coordinates becomes really similar as using the RGI as the following code example shows.

```py
    rbfi = RBFInterpolator(rz_points, data, **kwargs)
    data_fine = rgi(rz_points_fine)
```

The only difference is, that the data points are mapped to their corresponding poloidal coordinates and also are evaluated at the fine R-Z-positions.

Normally, extrapolating is really simple with the RBFI, as the generated functions can be evaluated at any point, even outside the grid. 
However, neither the fourier coefficients $hat(f)$ nor the $zeta$-shift are periodic in $s$.

Extending the grid using parallel periodic boundary conditions as done with interpolation in hamada coordinates is also not an option.
In hamada interpolation the s-grid is extended regularly outside of $-0.5<s<0.5$.
But, because $s$ is periodic ($s=s±1$), corresponding extended poloidal points would be coincide precisely poloidally.
This causes ambiguous data as different values are defined on the same poloidal points.
The RBFI cannot handle this and will throw a `SingularMatrix` Error.
Because the grid is discrete, its also not possible to add new points at the half-way point between $0.5-(Delta s)/2 < s < 0.5$, as the periodic points would be out of bounds as well. 
Therefore the interpolation results at the boundary will show numerical artifacts for poloidal interpolation.

===== Results and Comparison
To check whether the two interpolations methods give accurate results, the same GKW simulation was conducted with a low and a high resolution grid. 
The low resolution data was then upscaled by each interpolator to match the fine grid.
All plots in this section were created using the new regular triangulation method.

For an intuitive comparison of the interpolated potential $Phi'$ versus the potential of the accurate simulation $Phi$, the relative (normalized) difference of the two is plotted via the following formula:

$ Delta = frac(abs(Phi' - Phi), max(Phi)) . $

====== Circular

#include "../../figs/compare_interpolation/lin/circ/fig.typ"

One can immediately notice the strong deviation at the left side of the plot in the RBFI results.
It's located more precisely between the first in the gap of the $s$-grid as specified in @eq:discrete_s.
The RGI overall performs better than the RBFI.

#figure(
    table(
        columns: 3,
        [*difference in %*], [*max*], [*mean*],
        [*RGI*], [0.23], [0.02],
        [*RBFI*], [0.73], [0.03]
    ),
    caption: [Relative differences for $N_s=32$ to $N_s=128$ circular interpolation.]
)

====== CHEASE

The same was done with two CHEASE boundary simulations, one with $N_s=128$ the other with $N_s=512$.
Two subsections of the plot are analyzed, specifically the same ones used to benchmark triangulation in @sec:triang:limits.
First, a low density section around $s=plus.minus 0$ is analyzed.
The second one is a highly sheared section around $s=plus.minus 0.5$.
Not that neither figure is to scale, to better visualize the interpolation effects.

#include "../../figs/compare_interpolation/chease/sparse/original.typ"

The figure above shows the two simulation runs without subsequent interpolation.
The structure of the potential is very similar in both plots.
The low density plot is characterized by sharply defined edges, where the high density plot generates smooth curves.

#include "../../figs/compare_interpolation/chease/sparse/interpolation.typ"

The RBFI does not perform well in this section, where distances of points differ by multiple orders of magnitude.
It produces artifacts with relative differences of up to 98% in some sections.
The RGI on the other hand performs really well in this section and generates data that is really similar to the original plot.

#include "../../figs/compare_interpolation/chease/sheared/original.typ"

The low density plot shows an irregular pattern consisting of red, blue and white stripes, while the plot on the right is more nuanced.

#include "../../figs/compare_interpolation/chease/sheared/interpolation.typ"

In this highly sheared sections both interpolation methods yield similar results.
One can notice, that the lines are more "washed out" radially in the RBFI results.

The general interpolation benchmark can be subsumed by the table below.

#figure(
    table(
        columns: 3,
        [*difference in %*], [*max*], [*mean*],
        [*RGI*], [0.42], [0.08],
        [*RBFI*], [0.98], [0.11]
    ),
    caption: [Relative differences for $N_s=128$ to $N_s=512$ CHEASE interpolation.]
)

=== Non-linear Simulations
==== Functionality // TODO: Überschrift
The general process of calculating the potential on a poloidal slice in the non-linear case is described in detail in @sec:topovis:nonlin and can be subsumed as follows:

#enum(
    numbering: "1.",
    [Repeat $zeta$ and 3d-potential $Phi$ to the whole torus],
    [Calculate $zeta$-shift],
    [Interpolate $Phi$],
    [Evaluate potential $Phi$ at $zeta$-shift]
)

The process of upscaling $Phi$, would be no more than evaluating the splines it on the fine grid using the RGI.
But due to the specifications of the discrete grid discussed in @sec:discrete_grid, there is a gap in the $s$-grid where interpolation is not possible (see @eq:discrete_s).
To circumvent this, the grid has to be extended considering parallel periodic boundary conditions. 
Both $zeta$-shift and the potential $Phi(s,psi,zeta)$ have to be extended that way.

For $zeta$-shift the parallel periodic boundary condition (@eq:double_periodic_zeta) applies.

$ zeta(s) = zeta(s±1) ∓ q(psi) $

This translates to the potential as follows:

$ Phi(s,psi,zeta(s)) = Phi(s±1, psi, zeta(s±1) ∓ q(psi)) $

===== 1. Extend the sparse grid
As the RGI demands a strictly ascending or descending grid, the sparse s-grid has to be extended regularly _without_ accounting for boundary conditions, while $psi$ and $zeta$ are not modified. 
This extended grid is used to define the _virtual_ positions of the periodically extended potential.
The term virtual is chosen, because the points lie outside of domain $-0.5 < s < 0.5$ and are used as reference points for the RGI.

Additionally, the sparse $(s,psi,zeta)$ grid needs to be extended periodically in regard of the parallel periodic boundary conditions. 
The s-grid is periodic (see @eq:periodic:s) is therefore extended by wrapping the array periodically.
The periodic $zeta_p$-grid has to account for the parallel periodic boundary condition

$ zeta(s) = zeta(s±1) ∓ q $
// Gleichung rauslassen??

In code this is done by selectively adding or subtracting $q$ based on its $s$-coordinate.

```py
    sss_p, xxx_p, zzz_p = np.meshgrid(s_p, x_p, z_p, indexing='ij')
    zzz_p[-OVERLAP:None, :, :] -= dat.q[None, :, None]  # s > 0.5
    zzz_p[:OVERLAP, :, :] += dat.q[None, :, None]       # s < -0.5
    zzz_p = zzz_p % np.max(z)                           # inaccurate!
```

Where `s_p` denotes the periodically extended s-grid. 
To avoid mix-ups the `x_p` ($psi$) and `z_p` ($zeta$) arrays are also denoted that way, even though they are not extended in any way. 
The dimensionality of the safety factor `q` is extended to correctly add or subtract it from $zeta$ depending on its $psi$ value.
Note that instead of mapping $zeta_p$ back to $[0,1]$, it gets mapped to $[0,max(zeta)]$.
This is currently a workaround to avoid out-of-bounds errors when evaluating the potential, which is only defined in the range of $zeta$. 
The error for this is small as $max(zeta) approx 1$, but it's something that should be worked on in future iterations of ToPoVis.

===== 2. Extend the potential $Phi$
The potential $Phi$ must fulfill the following parallel periodic boundary condition:

$ Phi(s,psi,zeta(s)) = Phi(s±1, psi, zeta(s±1) ∓ q(psi)) $

Doing so requires prior interpolation.
This is because $Phi$ needs to be evaluated at the parallel wrapped $zeta_p$ (not $zeta$-shift!) positions, which in general do not coincide with the original $zeta$ positions.
In code this is done by initializing the interpolator with the sparse potential and grid and then calling it multiple times to evaluate on the extended grids.

```py
    pot3d_rgi = RegularGridInterpolator((s, x, z), whole_pot, **kwargs)
    # initialize extended periodic potential
    pot3d_p = np.zeros((len(s_e), len(x_e), len(z_e)))

    # s > 0.5
    slc = np.s_[-OVERLAP:None, :, :]
    points = grid_to_points((sss_p[slc], xxx_p[slc], zzz_p[slc]))
    p = pot3d_rgi(points)
    pot3d_p[slc] = np.reshape(p, shape=(np.shape(pot3d_p[slc])))

```

The exact same is done again for $s < -0.5$ using `slc = np.s_[:OVERLAP, :, :]`.
The preparation for this step is already done constructing the periodic $(s,psi,zeta)$-grid, as the potential is just evaluated at these coordinates and inserted in the corresponding position into the `pot3d_p` array.
As the RGI proofed successful in linear simulations its the only interpolator used in the non-linear cases.

===== 3. Extending $zeta$-shift
For $zeta$-shift the same boundary condition applies as for $zeta$:

$ zeta_s (s) = zeta_s (s±1) ∓ q $

===== 4. Interpolating extended $zeta$'\-shift
After $zeta$\-shift is extended using its parallel periodic boundary condition, it needs to be interpolated to the fine #sym.psi\-s-grid.
This is done using the RGI.

===== 5. Interpolating extended potential $Phi$'
Most arrays in ToPoVis like $hat(f)$, $zeta_s$, $R$ and $Z$ are in the shape $(N_psi, N_s)$. 
Following this, the fine $psi$-$s$-grid is also defined after the scheme.
The 3-dimensional potential, however, is defined as $Phi(s, psi, zeta)$. 
So before interpolation can be done, the fine grid has to be put into the same shape as the potential.
In code this is done by expanding the $psi$-$s$-grid in the $zeta$ dimension and combining it with $zeta$\-shift as the third dimension.

```py
    sss_fine = np.expand_dims(ss_fine, -1)
    xxx_fine = np.expand_dims(xx_fine, -1)
    zzzeta_s_fine = np.expand_dims(zeta_s_fine, -1)
    sxz_fine = sss_fine, xxx_fine, zzzeta_s_fine
```

After evaluating $Phi'$ on the fine $s$-$psi$-$zeta$-grid, the $zeta$-dimension gets discarded and the potential gets transposed to be of shape $(N_(psi,text("fine")), N_(s,text("fine")))$ for consistency.

==== Results
In the case of non-linear simulations, comparison between different simulations is not possible.
This is due to the chaotic nature of non-linear simulations. 
Even two simulations using the exact same input parameters will eventually lead to different results.
Therefore the only way to test the accuracy of the interpolation is by _downsampling_ simulation data and then upscaling it again.

For this, a new feature is added to just use every nth $s$-point from every dataset.
The debug parameter `--dsf` controls how much the grid is sampled down, e.g. $"DSF"=4$ just keeps every forth $s$-grid point. 
Note, that this is an experimental feature, which is added for this sole purpose and is not tested extensively.
Downscaling of the $psi$-grid or $zeta$-grid is also not currently supported.

The test is done using a high-resolution ($N_s=64$) simulation of the well known _cyclone benchmark case_ (see @dimits2000simulations). 
The $s$-grid resolution is then downsampled by factors 2, 4 and 8 before upscaling it through interpolation by the same factor using cubic splines.
Each downsampled low-resolution data is plotted both with and without interpolation. For validation, each upscaled potential $Phi'$ is compared against the original high-resolution potential $Phi$ by calculating their relative difference using the following formula:

$ Delta = frac(abs(Phi' - Phi), max(Phi)) $

#include "../../figs/compare_interpolation/nonlin/fig.typ"

One can immediately notice, how the quality of the downsampled plots worsens as the grid resolution decreases.
The plots get smeared poloidally and the circular shape degenerates to the shape of polygons.
Meanwhile the upscaled results are very similar.
The circular shape defined by the $R$ and $Z$ grids are being reconstructed as possible through cubic interpolation.
When scaling the $s$-grid down by a factor of 2, upscaling is nearly lossless.
The relative differences increase when further reducing the resolution. 
In @tab:nonlin:diffs below, the max and mean relative differences are displayed.

#figure(
    table(
        columns: 3,
        [*differences in %*], [*max*], [*mean*],
        [*DSF2*], [$0.065$], [$< 0.001$],
        [*DSF4*], [$1.26$], [$0.12$],
        [*DSF8*], [$1.46$], [$0.21$],
        [*DSF16*], [$1.24$], [$0.20$]
    ),
    caption: [Relative differences of upscaled data and exact values],
) <tab:nonlin:diffs>

== Miscellaneous
The new ToPoVis code is completely rewritten from scratch, due to the profound changes, that are needed to implement interpolation into the original version of ToPoVis.
Doing so also allows for varies additional restructuring and optimizing of the code.
These changes in itself are to small to group into their own sections, so they are listed in this section.

===== Dropped FFT-Interpolation Support
The option to interpolate the potential using FFT in non-linear cases was not reimplemented in the new version of ToPoVis.
FFT interpolation yields worse results compared to spline interpolation as benchmarks show @samaniego2024topovis[p.42].

===== Updated gmap
As the resolution could be upscaled through interpolation, a discontinuity could be notices at $s=plus.minus 0.5$ in CHEASE geometry visualizations.
This is likely caused by an error in the calculation of $zeta$-shift, which involves a numerical integration as specified in @eq:G:chease.
To avoid the need for this numerical integration, the definition of `gmap` in GKW has been changed. 
This makes the calculation of $zeta$-shift independent of geometry.
The method `shift_zeta` is adjusted accordingly to always calculate $zeta$-shift using the general formula 

$ zeta_s = -phi/(2 pi) + G(psi, s) $

The function $G(psi,s)$ is calculated accordingly depending on the geometry by GKW during runtime and saved to `gmap`.
To provide backward compatibility, the flag `--legacy-gmap` is added.
Setting this restores legacy behavior to calculate $zeta$-shift depending on the geometry as specified in @sec:background:topovis:zeta-shift.

===== Optimizations
While the original version, solves several calculations iteratively using `for`-loops, the new code is optimized to use vectorization instead.
One example of this is the interpolation of the potential in the non-linear case, which is now done on a three-dimensional grid using the RGI.

Additionally, the two import functions for the real and imaginary fourier coefficients are combined into one to avoid reshaping `parallel.dat` multiple times.

===== Additional Arguments
The list of arguments to call ToPoVis with, is extended to allow for further customization.
The new arguments are grouped into two categories: 1) interpolation and 2) plotting.
A full list of arguments and new usage can be found in @chap:usage.

===== Restructuring and Readability
The main process is restructured to a proper `main(args)` function, which has several advantages.
It increases readability and helps differentiate variables and their visibility.
Furthermore, it allows ToPoVis to be imported as a local python package and called from within other scripts. 
The input parameters can then be supplied using the `args` parameter instead of the command line. 
The final data is returned by the function for further processing.

In an attempt to standardize the importing of data sets from the `gkwdata.h5` files, a new class `GKWData(path, poten_timestep)` is added.
Upon initialization, all necessary data sets are read from file, reformatted for easier access and saved as class variables.
This avoids the use of pseudo constants.

On the same note the class `ToPoVisData` is added as singular variable that is being returned, when called from within another script.
It stores all arguments and results of the code as class variables and includes a comprehensive plotting function.