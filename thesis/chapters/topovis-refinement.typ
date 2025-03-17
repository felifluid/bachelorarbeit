#import "../functions.typ" : load-bib
#import "@preview/subpar:0.2.1"

= Improving the ToPoVis Code

== Numerical artifacts
With the original version of ToPoVis some numerical artifacts could be observed as can be seen in @fig:artifacts:contour_artifacts. Fixing these artifacts is the original motivation underlying this thesis. // TODO: satz umformulieren

// @Flo important for his current research:
// Visualisierung, Rand von simulationen in CHEASE geometrie, v.a. in nicht-linearen simulationen

To create the plots in ToPoVis and the plots in this thesis the function `tricontourf` from the package `matplotlib.pyplot` is used @samaniego2024topovis[27].
As the name suggests it uses an unstructured triangle mesh to draw contour regions.
If no triangle grid is supplied, it will generate one implicitly using a delaunay algorithm @matplotlib2025tricontourf. 
As a contour plot is a form of interpolation, the result can vary heavily with the choice of triangulation. // !! satz korrigieren

@fig:artifacts shows a subsection of simulation data in CHEASE geometry outputted by ToPoVis. // TODO: add specific s values
The section stands out with a really low density in poloidal direction and a high density in the radial direction and can therefore be classified as a heavily non-uniform grid.
As explained in the previous section ??,//@sec:triang !! fix this
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
In this area the constant-s-lines curve counter-clockwise. Because of this 

// however it doesn't come without caveats: due to the poloidal shift 

// every triangulation is a form of interpolation

// as later experiments show: hamada is the prefered geometry to interpolate

// delaunay triangulation in general "would" be better, if it wouldn't lead to artifacts on a sparse non-uniform grid

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