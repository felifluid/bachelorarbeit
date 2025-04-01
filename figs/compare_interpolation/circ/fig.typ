#import "@preview/subpar:0.2.1"

#subpar.grid(
  columns: 3,
    figure(
        image("ns128-fs4-rbfi.png"),
        caption: "NS=32; FS=4; RBFI",
    ), <fig:interp:circ:rbfi:interp>,
    figure(
        image("ns128-fs1-rgi.png"),
        caption: "NS=128; original",
    ), <fig:interp:circ:rbfi:original>,
    figure(
        image("ns128-fs4-rbfi-diff.png"),
        caption: "difference (absolute)",
    ), <fig:interp:circ:rbfi:diffs>,
    caption: [ 
        RBF interpolation results. Upscaling from NS=32 to NS=128.
    ],
    label: <fig:interp:circ:rbfi>,
)