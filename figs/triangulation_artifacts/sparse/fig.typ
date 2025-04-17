#import "@preview/subpar:0.2.1"

// TODO: reduce margins & spacings

#subpar.grid(
    figure(
        image("delaunay_contour.png"),
        caption: "delaunay contour plot",
    ), <fig:artifacts:contour_artifacts> ,
    figure(
        image("delaunay_grid.svg"),
        caption: "delaunay triangulation",
    ), <fig:artifacts:delaunay_triangles>,
    figure(
        image("regular_contour.png"),
        caption: "regular contour plot",
    ), <fig:artifacts:contour_regular>,
    figure(
        image("regular_grid.svg"),
        caption: "regular triangulation",
    ), <fig:artifacts:regular_triangles>,
    columns: (1fr, 1fr),
    caption: [ Different plots of the same subsection ($R=[1.245,1.26], Z=[-0.02, 0.092]$) of a poloidal cross section characterized with a non-uniform grid structure at $s=±0$ in CHEASE geometry.
    ],
    label: <fig:artifacts>,
)