#import "../functions.typ" : load-bib

= Improving the ToPoVis Code

== Numerical artifacts
=== What is the cause of artifacts?
// one main issue: numerical artifacts caused by triangulation on non-uniform grid

// visualize this with an image

// this becomes more apparent if a scatter + triplot is used instead

// how can triangulation be improved?


=== Refining the grid through interpolation

// solution: interpolation of said grid

// interpolation happens on regular hamada grid

// interpolation results: picture comparision

=== Extending the grid through double-periodic boundary conditions

== Miscellaneous

// more efficient as calculations got more vectorized instead of through for loops

// more arguments to modify plot


#load-bib()