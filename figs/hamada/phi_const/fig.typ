#import "@preview/subpar:0.2.1"

#subpar.grid(
  columns: 2,
  figure(
    image("toroidal.png"),
    caption: [toroidal],
  ), <fig:hamada:phi:t>,
  figure(
    image("hamada.png"),
    caption: [hamada]
  ), <fig:hamada:phi:h>,
  caption: [
    Hamada and toroidal surfaces for $phi=text("const")$. The color gradient represents $zeta$.
  ],
  label: <fig:hamda:phi>,
)