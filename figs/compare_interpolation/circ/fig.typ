#import "@preview/subpar:0.2.1"

#subpar.grid(
  columns: 3,
    figure(
        image("ns32-fs1-rgi.png"),
        caption: $N_s=32$,
    ), <fig:interp:circ:ns32>,
    figure(
        image("ns128-fs1-rgi.png"),
        caption: $N_s=128$,
    ), <fig:interp:circ:ns128>,
    caption: [ 
        No interpolation.
    ],
    label: <fig:interp:circ>,
)