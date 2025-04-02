#import "@preview/subpar:0.2.1"

#subpar.grid(
  columns: 3,
    figure(
        image("ns128-fs4-rgi.png"),
        caption: "upscaled",
    ), <fig:interp:circ:rgi:upscaled>,
    figure(
        image("ns128-fs4-diff.png"),
        caption: "difference (absolute)",
    ), <fig:interp:circ:rgi:diffs>,
    caption: [ 
        RGI interpolation results.
    ],
    label: <fig:interp:circ:rgi>,
)