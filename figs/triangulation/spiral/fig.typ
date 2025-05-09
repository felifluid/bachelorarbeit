#import "@preview/subpar:0.2.1"

#subpar.grid(
  figure(
    image("lin.svg"),
    caption: [spiral]
  ), <fig:spiral:single_lin>,
  figure(
    image("noisy.svg"),
    caption: [noisy spiral],
  ), <fig:spiral:single_noisy>,
  figure(
    image("interpolation.svg"),
    caption: [double spiral],
  ), <fig:spiral:double_lin>,
  figure(
    image("interpolation_noisy.svg"),
    caption: [noisy double spiral],
  ), <fig:spiral:double_noisy>,
  columns: (0.8fr, 0.8fr),
  caption: [
    Comparison of delaunay triangulations for a two spiral distrubutions with and without noise.
  ],
  label: <fig:delaunay_spiral>,
)