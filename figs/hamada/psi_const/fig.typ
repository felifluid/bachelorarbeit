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
    Hamada and toroidal surfaces for $psi=text("const")$. The color gradient represents the #sym.zeta coordinate.
  ],
  label: <fig:hamda:x>,
)