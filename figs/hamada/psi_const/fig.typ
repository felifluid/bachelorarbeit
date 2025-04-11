#import "@preview/subpar:0.2.1"

#subpar.grid(
  columns: 2,
    figure(
    image("toroidal.png"),
    caption: [toroidal],
  ), <fig:hamada:x:t>,
  figure(
    image("hamada.png"),
    caption: [hamada]
  ), <fig:hamada:x:h>,
  caption: [
    Toroidal and hamada surfaces for $psi=psi_"max"$. The color gradient represents $zeta$.
  ],
  label: <fig:hamda:x>,
)