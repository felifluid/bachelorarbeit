#import "@preview/subpar:0.2.1"

#subpar.grid(
    figure(
        image("grid.png"),
        caption: "grid + triangulation",
    ), <fig:artifacts:sparse:delaunay:grid>,
    figure(
        image("contour.png"),
        caption: "resulting contour plot",
    ), <fig:artifacts:sparse:delaunay:contour> ,
    columns: (1fr, 1fr),
    caption: [ Different plots of the same subsection ($R=[1.245,1.26], Z=[-0.02, 0.092]$) of a poloidal cross section characterized with a non-uniform grid structure at $s=±0$ in CHEASE geometry.
    ],
    label: <fig:artifacts:sparse:delaunay>,
)