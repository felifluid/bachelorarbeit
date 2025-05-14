= Conclusion

The aim of this thesis was to look into and fix numerical artifacts, that occurred in some visualizations created by the previous ToPoVis code.
The cause of the artifacts was identified as unfavorable delaunay triangulation.
Simulations in CHEASE geometry with a highly non-uniform point distributions are particularly affected by this.
To explore means of fixing these artifacts, it was neccessary to understand the functionality of delaunay triangulation.

One attempt was made, by implementing a new triangulation method for creating regular grids.
However, this alone could not compensate the low density of the simulation data.
The artifacts could finally be diminished, by rewriting the ToPoVis code to support interpolating the simulation data.
With this, the likeliness of unfavorable triangulation is minimized, as the grid can be interpolated arbitrarily fine.
The interpolation process is implemented for both linear and non-linear simulation data.
Both interpolation in hamadian space using the RGI, as well as in poloidal space using the RBFI were explored.
As the functionality of the RBFI made it impossible to apply double periodic boundary conditions, hamadian interpolation was found out to perform better.
The results where validated by comparing them against computationally intensive high resolution simulations.
In the non-linear case, this isn't possible, because the plasma behaves as a chaotic system.
Therefore a high resolution simulation was downsampled and re-interpolated to the original resolution for validation purposes.
Overall, interpolation results only differ by a maximum of 1-2%, compared to the high resolution data even when increasing the grid size by a factor of four.
Additionally, some changes where made to improve the readability and usability of the code. 

In sum, these improvements made to the ToPoVis code help more accurately study complex plasma phenomena.
In particular they make it possible to study small scale turbulences at the edges of the tokamak in CHEASE geometry, an area that was most notably affected by numerical artifacts.

// TODO: where to find the code