#import "@preview/subpar:0.2.1"

#subpar.grid(
  figure(
    image("arbitrary.svg"), 
    caption: [
      arbitrary
    ]
  ), <fig:triangulation:scattered:arbitrary>,
  figure(
    image("delaunay.svg"), 
    caption: [
      delaunay
    ]
  ), <fig:triangulation:scattered:delaunay>,
  columns: (1fr, 1fr),
  caption: [
    Comparison of two triangulations of the same set of vertices
  ],
  label: <fig:triangulation:scattered>,
)