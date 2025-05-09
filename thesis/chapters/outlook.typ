= Outlook
While the changes made to the ToPoVis code could substantially improve visualizations, there are still some things that could be refined.
For one, the program hasn't been benchmarked for interpolating the $psi$-grid, as its resolution has to be fairly high for a simulation to converge.
It could be interesting, if interpolation still produces good results when both $s$- and $psi$-grid are really sparse.
Simulations with high density to begin with, didn't see a lot improvements through interpolation in limited testing.
So it is doubtful that interpolation can refine visualizations in those cases, even though that would have to be validated.

When it comes to triangulation further enhancements could be made.
The regular triangulation, while providing predictable meshes, is no optimal solution for heavily non-uniform grid as present in CHEASE geometry.
However, there is a possible solution for this called _Delaunay Refinement_. 
Projects like _Tinfour_ by #cite(<lucas2018delaunay>, form: "prose") or _Triangle_ by #cite(<shewchuck2025triangle>, form: "prose") provide algorithms and packages for delaunay mesh refinement.
Practically, they work by specifying a lower bound for every angle or a maximum area.
The triangulation is then refined until all triangles fulfill these conditions.
For this to work, new grid points have to be created at arbitrary points within the convex hull of the mesh, which is possible through interpolation.

In future iterations, runtime improvements could be made for interpolation in non-linear cases. 
The process currently uses two three-dimensional spline interpolations.
The construction of the splines is very computational intensive, which has a big impact on runtime. 
This could possibly be improved by interpolating the grid _before_ extending the grid to the whole torus, or by reusing parts of the splines for the second interpolation.

As the interpolation proofed to be resilient, some hopes can be raised to save on extensive computations by simulating on low density grids.
However, this might proof difficult.
A certain amount of grid points are needed for the simulation to converge with time. 
Additionally, this approach might not work at all for non-linear simulations, as a different grid parameters can substantially change the results of the simulation making them incomparable.
Further benchmarking of the interpolation and cross-validation of the results is needed.