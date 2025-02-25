#import "@preview/subpar:0.2.1"
#import "../functions.typ" : load-bib

= Background
== GKW
The _Gyrokinetic Workshop (GKW)_ is a code to simulate and study turbolences of a confined plasma, usually inside of a tokamak @peeters2015[p.1]. It is written in Fortran 95 and was initially developed at the University of Warwick in 2007 @peeters2015[p.1]. The code is freely availiable and currently being hosted at https://bitbucket.org/gkw/gkw. 

== Hamada coordinates

// image of a tokamak torus with hamada coordinates

== Triangulation

In the context of graph-theory or computational geometry _triangulation_ is the maximum set of edges for a given set of vertices so that no edges are intersecting @klein2005voronoi[p.233]. Given a set of points, there are many possible ways of performing a triangulation. However, not all triangles are created equal. // TODO: informal, is this okay?

#subpar.grid(
  figure(
    image("../../figs/random_triangles.svg"), 
    caption: [
      arbitrary
    ]
  ), <arbitrary_triangulation>,
  figure(
    image("../../figs/delaunay_triangles.svg"), 
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

@arbitrary_triangulation above illustrates, that arbitrary triangulations can lead to many acute triangles. This is undesirable in many cases, as it can lead to numerical artifacts @klein2005voronoi[p.234]. 
The triangles in @delaunay_triangulation on the other hand tend to be more equilateral leading to a more uniform representation of the grid @lucas2018delaunay. The _delaunay triangulation_ achieves this by maximizing the minimal interior angle of all triangles @klein2005voronoi[p.234]. // ??: add further information? e.g. the circumcircle criterion?

// But there are also limits to the delaunay triangulation, which comes from two assumptions. No subset of four points are on the same circumcircle and no subset of three points lie on a straight line @klein2005voronoi[p.234]. 

While delaunay triangulation works efficiently when the grid is more or less uniform, it doesn't perform well on non-uniform grids @lo2013multigrid[p.15]. // TODO: kinda misleading → cpu performance is not great under certain circumstances
Moreover, different examples from #cite(<lo2013multigrid>,form: "prose", supplement: [p.21]), #cite(<peethambaran2015delaunay>, form: "prose", supplement: [p.166ff]) and #cite(<liu2008delaunay>, form: "prose", supplement: [p.1269]) show, that delaunay triangulation on non-uniform grids leads to many acute or big triangles. This can be illustrized well by examining the delaunay triangulation results of a spiral distrubution.

// ??: are these images too small?
#subpar.grid(
  figure(
    image("../../figs/triangulation/spiral_lin.svg"),
    caption: [spiral]
  ), <fig:spiral_linear>,
  figure(
    image("../../figs/triangulation/spiral_noisy.svg"),
    caption: [noisy spiral],
  ), <fig:spiral_noisy>,
  figure(
    image("../../figs/triangulation/spiral_interpolation.svg"),
    caption: [two noisy spirals],
  ), <fig:two_noisy_spirals>,
  columns: (1fr, 1fr, 1fr),
  caption: [
    Comparison of delaunay triangulations for three spiral distrubutions.
  ],
  label: <fig:delaunay_spiral>,
)

Even on a non-uniform grid like the spiral shown in @fig:spiral_linear, the delaunay triangulation results in a uniform representation of the grid. However, this triangulation is quiet sensitive to noise, as can be observed in @fig:spiral_noisy. This is caused by preffering tiny, but delaunay-conform triangles in high-density ares, instead of big but acute triangles in low-density areas. // TODO: what excactly is caused?
A possible refinement of the triangulation is to add more data points in areas with low density, as being shown in @fig:two_noisy_spirals #footnote[#cite(<ruppert1995delaunay>, form: "prose") and further #cite(<shewchuk1996triangle>, form: "prose") present more elaborate ways of refining a triangulation. Both offer algorithmical approaches to add the least amount of extra vertices to the grid so that no resulting angles exceed a given angle. _Triangle_ is an implementation of this and is freely availiable at https://www.cs.cmu.edu/~quake/triangle.html @shewchuck2025triangle.]. // ??: maybe move this to outlook? 

However, in the context of three-dimensional data, every point is assigned a value. In order to add more additional points interpolation is necessary. // for example with height data

== Interpolation
// focus on multidimensional data, at this is the data dimensions being interpolated in topovis
When faced with multidimensional data, there are multiple different interpolation approaches to choose from. In this section two different methods are being introduced and discussed.

=== RegularGridInterpolator
The `RegularGridInterpolator` (RGI) is a python class provided by SciPy @scipy2025rgi. This class makes a few assumptions of the grid structure to avoid expensive triangulation and therefore speed up the interpolation @scipy2025rgi. That is, the grid must be rectilinear, means rectangular with even or uneven spacing @scipy2025rgi. The RGI supports different methods for interpolation, among them nearest, linear, cubic and quintic @scipy2025rgi. The last two involve solving a large sparse linear system @scipy2025rgi.

// TODO: explain trilinear interpolation?

// ??: different headline?
=== RBFInterpolator
Similar to the RGI The `RBFInterpolator` (RBFI) is also a class provided by SciPy @scipy2025rbf. 

// TODO: cite underlying papers

== ToPoVis
=== What is ToPoVis?
_ToPoVis_ is a python script developed by Sofia Samaniego in 2024 @samaniego2024topovis. It aims to compute and visualize poloidal cross sections of eletrostatic potential #sym.Phi inside a tokamak, hence the name "ToPoVis" (#strong[To]kamak #strong[Po]loidal cross section #strong[Vis]ualisation) @samaniego2024topovis[p.10,72]. ToPoVis works with the simulation data outputted by GKW.


// TODO: Output Bild von ToPoVis einfügen

// both linear and non-linear simulations
// geometries: circular, s-alpha, chease-global

=== How does ToPoVis work?

// zeta shift - how does it work

==== Linear simulations

==== Non-linear simulations


== What needs to be improved?

#load-bib()