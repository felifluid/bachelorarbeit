#import "@preview/subpar:0.2.1"

#subpar.grid(
  columns: 2,
    figure(
    image("toroidal.png"),
    caption: [cartesian],
  ), <fig:hamada:x:t>,
  figure(
    image("hamada.png"),
    caption: [hamada]
  ), <fig:hamada:x:h>,
  caption: [
    $psi=psi_"max"$ surfaces. The color gradient represents $zeta$.
  ],
  label: <fig:hamda:x>,
)