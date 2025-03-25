#import "@preview/subpar:0.2.1"

#subpar.grid(
  figure(
    image("../../placeholder.svg"), 
    caption: [
      arbitrary
    ]
  ), <arbitrary_triangulation>,
  figure(
    image("../../placeholder.svg"), 
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