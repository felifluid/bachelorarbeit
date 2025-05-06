#import "@preview/subpar:0.2.1"

#subpar.grid(
  columns: 3,
    figure(
        image("dsf2-fs1.png"),
        caption: "no interpolation",
    ), <fig:interp:nonlin:dsf2:original>,
    figure(
        image("dsf2-fs2.png"),
        caption: "upscaled",
    ), <fig:interp:nonlin:dsf2:upscaled>,
    figure(
        image("dsf2-fs2-diff.png"),
        caption: "difference to original",
    ), <fig:interp:nonlin:dsf2:diffs>,
    caption: [ 
        Downscaling factor: 2.
    ],
    label: <fig:interp:nonlin:dsf2>,
)

#subpar.grid(
  columns: 3,
    figure(
        image("dsf4-fs1.png"),
        caption: "no interpolation",
    ), <fig:interp:nonlin:dsf4:original>,
    figure(
        image("dsf4-fs4.png"),
        caption: "upscaled",
    ), <fig:interp:nonlin:dsf4:upscaled>,
    figure(
        image("dsf4-fs4-diff.png"),
        caption: "difference to original",
    ), <fig:interp:nonlin:dsf4:diffs>,
    caption: [ 
        Downscaling factor: 4.
    ],
    label: <fig:interp:nonlin:dsf4>,
)

#subpar.grid(
  columns: 3,
    figure(
        image("dsf8-fs1.png"),
        caption: "no interpolation",
    ), <fig:interp:nonlin:dsf8:original>,
    figure(
        image("dsf8-fs8.png"),
        caption: "upscaled",
    ), <fig:interp:nonlin:dsf8:upscaled>,
    figure(
        image("dsf8-fs8-diff.png"),
        caption: "difference to original",
    ), <fig:interp:nonlin:dsf8:diffs>,
    caption: [ 
        Downscaling factor: 8.
    ],
    label: <fig:interp:nonlin:dsf8>,
)