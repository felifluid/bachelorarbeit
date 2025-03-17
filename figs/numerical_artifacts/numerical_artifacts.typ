#import "@preview/subpar:0.2.1"

// TODO: reduce margins & spacings

#subpar.grid(
    figure(
        image("chease_numerical_artifacts.svg"),
        caption: "delaunay contour plot",
    ), <fig:artifacts:contour_artifacts> ,
    figure(
        image("chease_bad_triangles.svg"),
        caption: "delaunay triangulation",
    ), <fig:artifacts:delaunay_triangles>,
    figure(
        image("chease_no_artifacts.svg"),
        caption: "regular contour plot",
    ), <fig:artifacts:contour_regular>,
    figure(
        image("chease_regular_triangles.svg"),
        caption: "regular triangulation",
    ), <fig:artifacts:regular_triangles>,
    columns: (1fr, 1fr),
    caption: [ Different plots of the same subsection ($R=[1.245,1.26], Z=[-0.02, 0.092]$) of a poloidal cross section in CHEASE geometry simulated by GKW.
    ],
    label: <fig:artifacts>,
)