#import "../functions.typ" : load-bib
#import "@preview/subpar:0.2.1"

= Improving the ToPoVis Code

== Numerical artifacts
With the original version of ToPoVis some numerical artifacts could be observed as can be seen in @fig:artifacts:contour_artifacts. Fixing these artifacts is the original motivation underlying this thesis. // TODO: satz umformulieren

// @Flo important for his current research:
// Visualisierung, Rand von simulationen in CHEASE geometrie, v.a. in nicht-linearen simulationen

The artifacts are caused by an unfavorable delaunay triangulation done implicitly by the `pyplot.tricontourf` method. 
As explained in the previous section ?,//@sec:triang // !! fix this
heavily non-uniform distributions can lead to so called "fat" triangles in areas of low density, as well as many strongly acute triangles surrounding them. This is excactly what can be observed in @fig:artifacts:delaunay_triangles.

Note that @fig:artifacts shows a specific small part of the poloidal slice at which the numerical artifacts are the strongest. To further help visualizing this effect the axes aren't equally scaled. To make the triangles distinguishable in the bottom two plots only every fourth point in the #sym.psi - direction is used, which has very little influence on the triangulation.

#include "../../figs/numerical_artifacts/numerical_artifacts.typ"

A possible solution to this is presented in the two figures on the right. Instead of relying on complex algorithms to perform a delaunay-conform triangulation, a custom regular triangle grid is used. This is achieved by a new method called `make_regular_triangles`. 

The method makes a triangulation in the equirectangular (meaning equally spaced and rectangular) hamada coordinates. Both poloidal coordinates R and Z are exported by GKW as functions of #sym.psi and s, represented as discrete values in a 2d-array. Therefore, each poloidal point P is assigned two hamadian indices. Its possible to map the indices of the arrays to a set of regular triangles. Regular meaning triangles have the form $ #sym.sum _(i=0)^n [(i,i), (i+1, i), (i, i+1)] $ // ??: ask Leo how to correctly write this mathematically

As @fig:artifacts shows, the regular triangulation leads to a more uniform representation of the triangle mesh as well as the resulting contour plot. The approach guarantees that triangle sizes are within the same order of magnitude and have similar shapes. However the regular triangulation doesn't come without its own caveats. // ?? Fachbegriff für die Verwirbelung / Verschraubung der poloidalen coordinaten

// however it doesn't come without caveats: due to the poloidal shift 

=== Refining the grid through interpolation

// solution: interpolation of said grid

// interpolation happens on regular hamada grid

// interpolation results: picture comparision

=== Extending the grid through double-periodic boundary conditions

== Miscellaneous

// more efficient as calculations got more vectorized instead of through for loops

// more arguments to modify plot


#load-bib()