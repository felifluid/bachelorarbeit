#import "@preview/subpar:0.2.1"
#import "thesis/bib.typ" : load-bib

#set par(justify: true)

= Theoretical Background
== Triangulation

In the context of graph-theory or computational geometry _triangulation_ is the maximum set of edges for a given set of vertices so that no edges are intersecting @klein2005voronoi[p.233]. Given a set of points, there are many possible ways of performing a triangulation. However, not all triangles are created equal. // TODO: informal, is this okay?

#subpar.grid(
  figure(
    image("figs/random_triangles.svg"), 
    caption: [
      arbitrary
    ]
  ), <arbitrary_triangulation>,
  figure(
    image("figs/delaunay_triangles.svg"), 
    caption: [
      delaunay
    ]
  ), <delaunay_triangulation>,
  columns: (1fr, 1fr),
  caption: [
    Comparison of two triangulations of the same set of points
  ],
  label: <triangulation_comparison>,
)

Above in @triangulation_comparison above illustrates, that arbitrary triangulations can lead to many acute triangles. This is undesirable in many cases, as it can lead to numerical artifacts @klein2005voronoi[p.234]. 
The triangle grid in figure b) on the other hand tend to be more equilateral leading to a more uniform representation of the grid @lucas2018delaunay. The _delaunay triangulation_ achieves this by maximizing the minimal interior angle of all triangles @klein2005voronoi[p.234]. // ??: add further information? e.g. the circumcircle criterion?

But there are also limits to the delaunay triangulation, which comes from two assumptions. No subset of four points are on the same circumcircle and no subset of three points lie on a straight line @klein2005voronoi[p.234]. 


// Delaunay triangulation works best on uniform grids (source needed!).

// Refinement algorithms needs arbitrary grid points, therefore interpolation

// if we need interpolation anyways, we can interpolate the grid to a point, where we either don't need refinement of triangles (triangulation performs well), or regular grid works similarly well

// → refinement of grid isn't necessary

// überleitung zu interpolation

== Interpolation

// interpolation is the best option to refine the usually ununiform grids, which is needed for triangulation

// there are different approaches to interpolation.

// grids with different properties "need" different interpolation methods. (some methods work better, some methods won't work for some kinds of grids)

// for regular n-dimensional *regular* grids there is the RegularGridInterpolator from scipy. 

// it utilizes the regular characteristic of the grid data to *efficiently* construct splines. (this involves solving a sparse linear system) (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html#scipy.interpolate.RegularGridInterpolator)

// ??: do I add visualization & formulas of how *excactly* this interpolation works?

// HOWEVER, this can NOT extrapolate > other workaround have to be found

// ??: do I even include this? maybe just a short mention

// on an irregular (or scattered) grid other interpolation methods have to be used

// in testing the RBFInterpolator showed the best results for this matter

// each data point will get approximated as a radial base function

// each evaluation point involves evaluating the radial basis functions of n of its nearest neighbors

// memory usage scales quadrically with number of neighbors to consider for each evaluation point -> becomes inpractical for high numbers

// this approach has much similarities with the interpolation done implicitly by the triangulation.

// compare Voronoi-diagram with radial basis function

// however this approach is really computationally expensive

// should only be used, when data is only known on scattered grid

#load-bib()