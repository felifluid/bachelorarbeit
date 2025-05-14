#import "@preview/subpar:0.2.1"

// TODO: reduce margins & spacings

#subpar.grid(
    figure(
        image("delaunay/contour.png"),
        caption: "delaunay contour plot",
    ), <fig:sheared_triang:delaunay_contour> ,
    figure(
        image("delaunay/grid.png"),
        caption: "delaunay triangulation",
    ), <fig:sheared_triang:delaunay_triangles>,
    figure(
        image("regular/contour.png"),
        caption: "regular contour plot",
    ), <fig:sheared_triang:regular_contour>,
    figure(
        image("regular/grid.png"),
        caption: "regular triangulation",
    ), <fig:sheared_triang:regular_triangles>,
    columns: (1fr, 1fr),
    caption: [ 
        Four plots showing the effects of the triangulation on the contour plot. The presented subsections are characterized by a high poloidal shearing of the grid.
    ],
    label: <fig:sheared_triang>,
)