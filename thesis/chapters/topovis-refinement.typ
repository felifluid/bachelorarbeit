#import "../functions.typ" : load-bib
#import "@preview/subpar:0.2.1"

= Improving the ToPoVis Code

== Numerical artifacts
=== Cause of the Artifacts
With the original version of ToPoVis some numerical artifacts could be observed in the contour plots as can be seen in @fig:artifacts:contour_artifacts. 
The artifacts are particularly severe at the edges of plots in CHEASE geometry. 
This made it harder to study small scale turbulences in these areas.
Fixing these artifacts is the original motivation underlying this thesis. 

To create the plots in ToPoVis and the plots in this thesis the function `tricontourf` from the package `matplotlib.pyplot` is used @samaniego2024topovis[p.27].
As the name suggests it uses an unstructured triangle mesh to draw contour regions.
If no triangle grid is supplied, it will generate one implicitly using a delaunay algorithm @matplotlib2025tricontourf. 
The generation of smooth surfaces using triangulation is commonly used as a linear interpolation of (un-)structured three-dimensional data @lawson1977software[p.1-2]. 
The triangulation defines sets of 3 grid points, that will be interpolated inbetween. 
As such, the choice of triangulation has a big impact on the resulting interpolation. 
As discussed in section !!, delaunay triangulation usually yields the best results, however, it faces issues with heavily non-uniform grids.

@fig:artifacts shows a subsection of simulation data in CHEASE geometry outputted by ToPoVis. 
The section stands out with a really low density in poloidal direction and a high density in the radial direction and can therefore be classified as a heavily non-uniform grid.
As explained in the previous section !!,
non-uniform distributions can lead to so called "fat" triangles in areas of low density, as well as many strongly acute triangles surrounding them.
This is excactly what can be observed in @fig:artifacts:delaunay_triangles.
Note that the axes aren't scaled equally to help visualize this effect. 
Furthermore only every fourth point in the #sym.psi - direction is used to make the triangles distinguishable. 
This has very little influence on the delaunay triangulation in this specific section.

#include "../../figs/triangulation_artifacts/sparse/fig.typ"

A possible solution to this is presented in the bottom two figures. Instead of relying on delaunay triangulation, a custom regular triangle grid is used. This is achieved by a new method called `make_regular_triangles`. 

The method makes a triangulation in the equirectangular (meaning equally spaced and rectangular) hamada coordinates. Both poloidal coordinates R and Z are exported by GKW as functions of #sym.psi and s, represented as discrete values in a 2d-array. Therefore, each pair of indices in hamada describe a point P. Its possible to map the indices of the arrays to a set of regular triangles. Regular meaning triangles have the form $ #sym.sum _(i=0)^n [(i,i), (i+1, i), (i, i+1)] $ // ??: ask Leo how to correctly write this mathematically

While the function provides a better triangulation in non-uniform areas, it leads to many acute triangles in other areas. // TODO: specify area?
The CHEASE geometry is sheared poloidal in #sym.theta along the radial coordinate #sym.psi. However, the shearing is asymmetrical in the poloidal coordinate s. Around $s=0$ the shearing is minimal, so lines with constant s are nearly straight radially. The shearing reaches its maximum at $s=±0.5$. 
In this area the constant-s-lines curve counter-clockwise. Because of this several s-contant-lines span poloidal parallel to each other. This leads to a high density unstructured grid, which is favorable for delaunay triangulation.

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

A more and refined way to improve triangulations is to avoid the causes of unfavorable triangulations in the first place. Is can be achieved by interpolating the grid by other means _before_ triangulation is done, which will be discussed next.

// the best method would probably be to use delaunay refinement in combination with interpolation -> outlook

// of course simulations with higher density in s would help. but those are really heavy to compute. for convergence of the simulation only relatively few s grid points are needed.

=== Refining the Grid through Interpolation

Similarly to the triangulation, the interpolation of the grid can be done in both poloidal coordinates and hamada coordinates. In this chapter, each approach is tested on circular simulations with a low density s-grid ($N_s=32$) and compared with a simulation with four times the amount of s-grid points ($N_s=128$). The technical functionality of the two interpolation methods is discussed in detail in !!. //@sec:background:interpolation

// TODO: adjust this outlook

==== Linear Simulation
The potential on a poloidal slice in the linear case is calculated by first calculating #sym.zeta for $phi = #text("const")$ - known as #sym.zeta\-shift. 
The potential is then calculated using the #sym.zeta\-shift and the complex fourier coefficients using the formula

$ Phi(hat(f), zeta, k) = hat(f) dot e^(i k zeta) + hat(f)^* dot e^(-i k zeta) $

This leaves open two different strategies for interpolating the potential:

#set enum(numbering: "A)")
+ Calculate the potential on the sparse grid and interpolate it.
+ First interpolate #sym.zeta\-shift and $hat(f)$ and then calculate the potential on the fine grid.

While option A) is the simpler approach, it also generates worse results. 
This is presumably due to the potential being a real number.
So before interpolation the imaginary part of the complex fourier coefficients gets discarted.
Whereas the imaginary part is being kept and interpolated in option B).
Additionaly, $hat(f)$ and #sym.zeta\-shift each represent smoother surfaces than the potential, which makes interpolation less prone to numerical errors.
This makes option B) the preffered approach, while option A) will not be discussed or used further in this thesis.

// The main issue that comes with interpolating #sym.zeta and $hat(f)$ is, that both quantities are not continous at $s=±0.5$. 
// Meaning we cannot simply interpolate between this boundary, but instead have to apply certain boundary conditions to extend the grid out of bounds. 
// This topic is discussed in detail in section !!.

There is still the choice of the coordinate system to interpolate in. 
Both poloidal and hamadian interpolation will be discussed in the next sections.

===== Hamadian Interpolation
Interpolation in hamada coordinates is done using `RegularGridInterpolator`, which functionality is discussed in Section !!. 
Both #sym.zeta\-shift and the fourier coefficients $hat(f)$ are defined as 2d-arrays as functions of #sym.psi and $s$.
Therefore interpolation is a simple act of first initializing the interpolator with the sparse grid and data and then calling it to evaluate on the fine grid.

```py
rgi = RegularGridInterpolator((x, s), data, **kwargs)
data_fine = rgi(xs_points_fine)
```
The main restriction with this method is, that the RGI can only _interpolate_. 
This wouldn't be an issue, if the s-grid is defined as $s=[-0.5, 0.5]$. 
But due to the functionality of GKW, it is defined from $s_0=-0.5 + (Delta s) / 2$ to $s_(-1) = 0.5 - (Delta s)/2$. 
This leads to a gap of exactly $Delta s$ between $s_0$ and $s_(-1)$.
In order to make interpolation work in this gap, the grid has to be extended outside the bounds of $-0.5 < s < 0.5$. 
This is done by applying parallel periodic boundary conditions to the grid and the data arrays #sym.zeta\-shift and $hat(f)$.

In itself $s$ is perfectly periodic, meaning $s=s±1$. 
However, the RGI depends on a strictly ascending or descending grid. 
So instead of wrapping the s-grid periodically, it needs to be extended out of bounds with regular spacing. 
In code this is achieved by the method `extend_regular_array(a: arr, n: int)`, which first checks if the given array is equally spaced, and then extends it by `n` in both directions. 
The extended s-grid is defined from $s_0=-0.5-n dot (Delta s)/2$ to $s_(-1) = 0.5 + n dot (Delta s)/2$. 
It is only used to define the _virtual_ position of the periodically extended data points for the RGI.

The parallel periodic boundary condition for #sym.zeta is defined as

$ zeta(s) = zeta(s±1) ∓ q $

and for the fourier coefficient $hat(f)$ a phase shift has to be applied

$ hat(f)(s) = hat(f)(s±1) dot e^(±i k q) $

with $q=q(psi)$ being the safety factor. 

// Add code examples??

After applying these parallel periodic boundary conditions, both #sym.zeta\-shift and $hat(f)$ are continuous across $s=±0.5$.
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

Using the RBFI, interpolating in poloidal coordinates becomes really similar as using the RGI as the following code example shows.

```py
rbfi = RBFInterpolator(rz_points, data, **kwargs)
data_fine = rgi(rz_points_fine)
```

The only difference is, that the data points are mapped to their corresponding poloidal coordinates and also are evaluated at the fine R-Z positions.

Normally, extrapolating is really simple with the RBFI, as the generated functions can be evaluated at any point, even outside the grid. 
However, neither the fourier coefficients nor the #sym.zeta\-shift are periodic in $s$.

Extending the grid as done with interpolation in hamada coordinates is also not an option.
In hamada interpolation the s-grid is extended regularly outside of $-0.5<s<0.5$.
But, because of $s$ is periodic ($s=s±1$), corresponding extended poloidal points would be coincide precisely poloidally.
This causes ambigous data as different values are defined on the same poloidal points.
The RBFI cannot handle this and will throw a `SingularMatrix` Error.
Because the grid is discrete its also not possible to add new points at the half-way point between $0.5-(Delta s)/2 < s < 0.5$, as the periodic points would be out of bounds as well. 
Therefore the interpolation results at the boundary will show numerical artifacts for poloidal interpolation.

===== Results and Comparison
To check whether the two interpolations methods give accurate results, the same GKW simulation was conducted with $N_s = 32$ and $N_s = 128$. 
The low resolution data was then upscaled by each interpolator to match the fine grid.

#include "../../figs/compare_interpolation/circ/fig.typ"
 
#include "../../figs/compare_interpolation/circ/rbfi/fig.typ"

#include "../../figs/compare_interpolation/circ/rgi/fig.typ"

One can immediately notice the strong deviation at the left side of the plot in the RBFI results (#ref(<fig:interp:circ:rbfi>)).
It's located more precisely between the first $s_0=-0.5+(Delta s)/2$ and last $s_(-1)=0.5-(Delta s)/2$ constant-$s$ lines.


It can be observed, that the RGI overall performs better than the RBFI.

// TODO: 1D graph with mean in psi?

=== Extending the grid through double-periodic boundary conditions // !! move to background 

// problem: regular grid interpolator can only interpolate

// s-grid is defined from $s=0.5-Delta s/2$ to $s=-0.5+Delta s/2$. Note how this gap increases with lower s-grid resolution.

// therefore we are left with a blank space without interpolated data between the first and last s.

// of course this gap could be filled otherwise, e.g. through linear interpolation. However, this would give a false confidence of how the potential looks like in that area.

// instead, when the flag '--periodic' is not supplied and the triangulation method is set to 'regular', ToPoVis will neither interpolate nor triangulate between $s=0.5-Delta s/2$ to $s=-0.5+Delta s/2$. This will lead to a white, blank space in that area.

// However, there is an option to generate additional gridpoints *outside* the domain. This is being done through _double-periodic boundary conditions_.

== Miscellaneous

// more efficient as calculations got more vectorized instead of through for loops

// more arguments to modify plot

// made importable and callable as a local package

#load-bib()