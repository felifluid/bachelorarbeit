#import "@preview/subpar:0.2.1"

#subpar.grid(
  figure(
    image("spiral_lin.svg"),
    caption: [spiral]
  ), <fig:spiral_linear>,
  figure(
    image("spiral_noisy.svg"),
    caption: [noisy spiral],
  ), <fig:spiral_noisy>,
  figure(
    image("spiral_interpolation.svg"),
    caption: [two noisy spirals],
  ), <fig:two_noisy_spirals>,
  columns: (1fr, 1fr, 1fr),
  caption: [
    Comparison of delaunay triangulations for three spiral distrubutions.
  ],
  label: <fig:delaunay_spiral>,
)