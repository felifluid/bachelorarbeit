#import "@preview/subpar:0.2.1"

#subpar.grid(
  columns: 2,
  figure(
    image("toroidal.png"),
    caption: [poloidal],
  ), <fig:hamada:phi:t>,
  figure(
    image("hamada.png"),
    caption: [hamada]
  ), <fig:hamada:phi:h>,
  caption: [
    $phi=text("const")$ surfaces. The color gradient represents $zeta$.
  ],
  label: <fig:hamda:phi>,
)