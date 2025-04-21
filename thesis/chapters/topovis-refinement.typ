#import "@preview/subpar:0.2.1"

= Improving the ToPoVis Code

// TODO: passt die position hier?
#include "../../figs/triangulation_artifacts/sparse/artefacts.typ"

== Dealing with Numerical Artifacts
=== The Cause of Numerical Artifacts

With the original version of ToPoVis some numerical artifacts could be observed in the contour plots as can be seen in @fig:numerical_artifacts. 
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

@fig:artifacts:sparse:delaunay shows a subsection of simulation data in CHEASE geometry outputted by ToPoVis. 
The section stands out with a really low density of the $s$-grid and a high density in the radial direction and can therefore be classified as a heavily non-uniform grid.
As explained in the previous @sec:triang,
non-uniform distributions can lead to so called "fat" triangles in areas of low density, as well as many strongly acute triangles surrounding them.
This is excactly what can be observed in @fig:artifacts:sparse:delaunay:grid.
Note that the axes aren't scaled equally to help visualize this effect. 
Furthermore only every fourth point in the  $psi$-direction is used to make the triangles more distinguishable. 
This has little to no influence on the delaunay triangulation in this specific section.

=== An alternative Triangulation
#include "../../figs/triangulation_artifacts/sparse/regular/fig.typ"

A possible solution to this is presented in @fig:artifacts:sparse:regular up top.
Instead of relying on delaunay triangulation, a custom _regular_ triangle grid is used.
To achieve this, triangulation must be done in the equidistant hamada coordinates instead.
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

$ k_(i j) = i + j dot n $

Therefore, @eq:triangles_tuple can be rewritten to

$ T = sum_(i=0)^(n-2) sum_(j=0)^(m-2) overbracket([k_(i j), k_(i+1, j), k_(i, j+1)], #sym.triangle.tl) + overbracket([k_(i+1, j), k_(i+1, j+1), k_(i, j+1)], triangle.stroked.br) $ <eq:triangles:flattened>

In the context of poloidal periodicity another row of triangles has to be added.
This is equivalent to adding the following term to the above formula

$ sum_(i=0)^(m-2) overbracket([k_(i, m-1), k_(i+1,m-1), k_(i, 0)],triangle.tl) + overbracket([k_(i+1, m-1), k_(i+1,0), k_(i, 0)], triangle.br) $

again assuming index $j$ describes the poloidal $s$ coordinate.
Doing so adds triangles between the first $j=0$ and last $j=m-1$ values for all radial indices $i$.

In code this is done by the new method `make_regular_triangles` via list comprehentions.

=== Limits of Triangulation

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
Findings in section !! show, that interpolations in hamada coordinates are coinciding better with the high resolution simulations.

A more and refined way to improve triangulations is to avoid the causes of unfavorable triangulations in the first place. 
Is can be achieved by interpolating the grid by other means _before_ triangulation is done, which will be discussed next.

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
So before interpolation the imaginary part of the complex fourier coefficients gets discarted.
Whereas the imaginary part is being kept and interpolated in option B).
Additionaly, $hat(f)$ and $zeta$\-shift each represent smoother surfaces than the potential, which makes interpolation less prone to numerical errors.
This makes option B) the preffered approach, while option A) will not be discussed or used further in this thesis.

There is still the choice of the coordinate system to interpolate in. 
Both poloidal and hamadian interpolation will be discussed in the next sections.

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
Intuitively interpolation in poloidal coordinates seems like a good approach, as the coordinates describe the real world euclidian space.
If two points are really close to each other in euclidian space, one can expect similar measurements at these points. // ??: gibts dafür ein Fachbegriff?
As the electric potential is smooth and continuous inside the tokamak, this property also applies to it. // TODO: dont like this sentence
Interpolating in hamada coordinates does not take the euclidian distance into account. 
Due to the non-linear transformation between the two coordinate systems, two points can be close to each other in hamadian space, but have a large distance in euclidian space.

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
This causes ambigous data as different values are defined on the same poloidal points.
The RBFI cannot handle this and will throw a `SingularMatrix` Error.
Because the grid is discrete, its also not possible to add new points at the half-way point between $0.5-(Delta s)/2 < s < 0.5$, as the periodic points would be out of bounds as well. 
Therefore the interpolation results at the boundary will show numerical artifacts for poloidal interpolation.

===== Results and Comparison
To check whether the two interpolations methods give accurate results, the same GKW simulation was conducted with $N_s = 32$ and $N_s = 128$. 
The low resolution data was then upscaled by each interpolator to match the fine grid.

#include "../../figs/compare_interpolation/circ/fig.typ"

#include "../../figs/compare_interpolation/circ/rbfi/fig.typ"

#include "../../figs/compare_interpolation/circ/rgi/fig.typ"

For an intuitive comparison of the interpolated potential $Phi'$ versus the potential of the accurate simulation $Phi$, the relative (normalized) difference of the two is plotted via the following formula:

$ Delta = frac(abs(Phi' - Phi), max(Phi)) $

One can immediately notice the strong deviation at the left side of the plot in the RBFI results (#ref(<fig:interp:circ:rbfi>)).
It's located more precisely between the first $s_0=-0.5+(Delta s)/2$ and last $s_(-1)=0.5-(Delta s)/2$ constant-$s$ lines.


It can be observed, that the RGI overall performs better than the RBFI.

// TODO: 1D graph with mean in psi?

=== Non-linear Simulations
==== Functionality // TODO: Überschrift
The general process of calculating the potential on a poloidal slice in the non-linear case is described in detail in @sec:topovis:nonlin and can be subsumized as follows:

#enum(
    numbering: "1.",
    [Repeat $zeta$ and 3d-potential $Phi$ to the whole torus],
    [Calculate $zeta$-shift],
    [Interpolate $Phi$],
    [Evaluate potential $Phi$ at $zeta$-shift]
)

The process of upscaling $Phi$, would be no more than evaluating the splines it on the fine grid using the RGI.
But doing so, a gap arises between $s_0 = -0.5+(Delta s)/2$ and $s_(-1) = 0.5 - (Delta s)/2$ where interpolation is not possible.
To circumvent this, the grid has to be extended considering parallel periodic boundary conditions. 
Both $zeta$-shift and the potential $Phi(s,psi,zeta)$ have to be extended that way.

For $zeta$-shift the parallel periodic boundary condition (@eq:double_periodic_zeta) applies.

$ zeta(s) = zeta(s±1) ∓ q(psi) $

This translates to the potential as follows:

$ Phi(s,psi,zeta(s)) = Phi(s±1, psi, zeta(s±1) ∓ q(psi)) $

===== 1. Extend the sparse grid
As the RGI demands a strictly ascending or descending grid, the sparse s-grid has to be extended regularly _without_ accounting for boundary conditions, while $psi$ and $zeta$ are not modified. 
This extended grid is used to define the _virtual_ positions of the periodically extended potential.
The term virtual is chosen, because the points lie outside of domain $-0.5 < s < 0.5$.

Additionaly, the sparse $(s,psi,zeta)$ grid needs to be extended periodically in regard of the parallel periodic boundary conditions. 
The s-grid is periodic, (see @eq:periodic:s)

$ s = s±1 $

// Gleichung rauslassen??

and is therefore extended by wrapping the array periodically.
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
The dimensionality of the safety factor `q` is extended to correctly add or substract it from $zeta$ depending on its $psi$ value.
Note that instead of mapping $zeta_p$ back to $[0,1]$, it gets mapped to $[0,max(zeta)]$.
This is currently a workaround to avoid out-of-bounds errors when evaluating the potential, which is only defined in the range of $zeta$. 
The error for this is small as $max(zeta) approx 1$, but it's something that should be worked on in future iterations of ToPoVis.

===== 2. Extend the potential $Phi$
The potential $Phi$ must fulfill the following parallel periodic boundary condition:

$ Phi(s,psi,zeta(s)) = Phi(s±1, psi, zeta(s±1) ∓ q(psi)) $

Doing so requires prior interpolation.
This is because $Phi$ needs to be evaluated at the parallely wrapped $zeta_p$ (not $zeta$-shift!) positions, which in general do not conincide with the original $zeta$ positions.
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
The preperation for this step is already done constructing the periodic $(s,psi,zeta)$-grid, as the potential is just evaluated at these coordinates and inserted in the corresponding position into the `pot3d_p` array.
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

After evaluating $Phi'$ on the fine $s$-$psi$-$zeta$-grid, the $zeta$-dimension gets discarted and the potential gets transposed to be of shape $(N_(psi,text("fine")), N_(s,text("fine")))$ for consistency.

==== Results

== Miscellaneous

// more efficient as calculations got more vectorized instead of through for loops

// more arguments to modify plot

// made importable and callable as a local package

// readability of the code

// vectorize / optimize some calculations
// e.g. interpolation, parallel.dat reshape

// make topovis callable and importable