#import "@preview/subpar:0.2.1"

#subpar.grid(
    columns: 2,
    figure(
        image("ns126.png"),
        caption: [$N_s = 128$],
    ), <fig:interpolation:chease:sheared:low> ,
    figure(
        image("ns512.png"),
        caption: [$N_s = 512$],
    ), <fig:interpolation:chease:sheared:high>,
    caption: [
        Plots without interpolation.
    ],
    label: <fig:interpolation:chease:sheared>
)