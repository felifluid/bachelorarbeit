#import "@preview/subpar:0.2.1"

#subpar.grid(
  columns: 3,
    figure(
        image("ns128-fs4-rbfi.png"),
        caption: "upscaled",
    ), <fig:interp:circ:rbfi:interp>,
    figure(
        image("ns128-fs4-diff.png"),
        caption: "relative difference",
    ), <fig:interp:circ:rbfi:diffs>,
    caption: [ 
        RBF interpolation results.
    ],
    label: <fig:interp:circ:rbfi>,
)