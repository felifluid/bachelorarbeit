#import "@preview/subpar:0.2.1"

#subpar.grid(
    figure(
        image("grid.svg"),
        caption: "grid + triangulation",
    ), <fig:artifacts:sparse:regular:grid>,
    figure(
        image("contour.png"),
        caption: "resulting contour plot",
    ), <fig:artifacts:sparse:regular:contour> ,
    columns: (1fr, 1fr),
    caption: [ Plots of the same subsection using regular triangulation.
    ],
    label: <fig:artifacts:sparse:regular>,
)