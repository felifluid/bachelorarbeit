#import "../functions.typ" : load-bib
#import "@preview/subpar:0.2.1"

= Improving the ToPoVis Code

== Numerical artifacts
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

#include "../../figs/numerical_artifacts/numerical_artifacts.typ"

A possible solution to this is presented in the bottom two figures. Instead of relying on delaunay triangulation, a custom regular triangle grid is used. This is achieved by a new method called `make_regular_triangles`. 

The method makes a triangulation in the equirectangular (meaning equally spaced and rectangular) hamada coordinates. Both poloidal coordinates R and Z are exported by GKW as functions of #sym.psi and s, represented as discrete values in a 2d-array. Therefore, each pair of indices in hamada describe a point P. Its possible to map the indices of the arrays to a set of regular triangles. Regular meaning triangles have the form $ #sym.sum _(i=0)^n [(i,i), (i+1, i), (i, i+1)] $ // ??: ask Leo how to correctly write this mathematically

While the function provides a better triangulation in non-uniform areas, it leads to many acute triangles in other areas. // TODO: specify area?
The CHEASE geometry is sheared poloidal in #sym.theta along the radial coordinate #sym.psi. However, the shearing is asymmetrical in the poloidal coordinate s. Around $s=0$ the shearing is minimal, so lines with constant s are nearly straight radially. The shearing reaches its maximum at $s=±0.5$. 
In this area the constant-s-lines curve counter-clockwise. Because of this several s-contant-lines span poloidal parallel to each other. This leads to a high density unstructured grid, which is favorable for delaunay triangulation.

#include "../../figs/numerical_artifacts/sheared_triang/fig.typ"

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

=== Refining the grid through interpolation

// solution: interpolation of said grid

// interpolation happens on regular hamada grid

// interpolation results: picture comparision

=== Extending the grid through double-periodic boundary conditions

== Miscellaneous

// more efficient as calculations got more vectorized instead of through for loops

// more arguments to modify plot


#load-bib()