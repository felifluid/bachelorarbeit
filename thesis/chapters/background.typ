#import "../functions.typ" : load-bib
#import "@preview/subpar:0.2.1"

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

Above in @triangulation_comparison above illustrates, that arbitrary triangulations can lead to many acute triangles. This is undesirable in many cases, as it can lead to numerical artifacts @klein2005voronoi[p.234]. 
The triangle grid in figure b) on the other hand tend to be more equilateral leading to a more uniform representation of the grid @lucas2018delaunay. The _delaunay triangulation_ achieves this by maximizing the minimal interior angle of all triangles @klein2005voronoi[p.234]. // ??: add further information? e.g. the circumcircle criterion?

// But there are also limits to the delaunay triangulation, which comes from two assumptions. No subset of four points are on the same circumcircle and no subset of three points lie on a straight line @klein2005voronoi[p.234]. 


While delaunay triangulation works efficiently when the grid is more or less uniform, it doesn't perform well on non-uniform grids @lo2013multigrid[p.15]. Moreover, different examples from #cite(<lo2013multigrid>,form: "prose", supplement: [p.21]), #cite(<peethambaran2015delaunay>, form: "prose", supplement: [p.166ff]) and #cite(<liu2008delaunay>, form: "prose", supplement: [p.1269]) show, that delaunay triangulation on non-uniform grid leads to many acute or "fat" triangles.



// Refinement algorithms needs arbitrary grid points, therefore interpolation

// if we need interpolation anyways, we can interpolate the grid to a point, where we either don't need refinement of triangles (triangulation performs well), or regular grid works similarly well

// → refinement of grid isn't necessary

// überleitung zu interpolation

== Interpolation

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