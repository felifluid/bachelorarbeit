#import "@preview/subpar:0.2.1"
#import "../functions.typ" : load-bib,  enum_numbering

// FIXME: this should be in preamble
#set heading(numbering: "1.")

= Background
== GKW
The _Gyrokinetic Workshop_ (GKW) is a code to simulate and study turbolences of a confined plasma, usually inside of a tokamak @peeters2015[1]. It is written in Fortran 95 and was initially developed at the University of Warwick in 2007 @peeters2015[1]. The code is freely availiable and currently being hosted at https://bitbucket.org/gkw/gkw. It works in both linear and non-linear regimes @peeters2015[1]

// linear case
// GKW solves eigenvalues k_zeta (fourier eigenmodes)
// eigenvalues are symmetrical
// as "linear" they do not couple and evolve independently
// usually only one fourier mode is simulated
// → fourier coefficient is 2D (psi, s) with k_zeta constant

// nonlinear run

// ?? add section for toroidal coordinates

== Hamada coordinates
To efficiently solve the gyrokinetic equation, GKW makes use of so called _hamada coordinates_.
Hamada coordinates are retrieved by transforming the toroidal coordinates in such a way that

+ field lines become straight and
+ one of the coordinates is aligned with the magnetic field. // Formatierung??

For further reading on how this is achieved in detail, see @peeters2015[23ff].

In circular geometry the coordinate transformation is defined by the following equations (see @peeters2015[A1]):

$
  &psi(r) = r/R_text("ref") \ 
  &s(r, theta) = 1/(2 pi) (theta + psi sin(theta)) \
  &zeta (r, theta, phi) = - phi/(2 pi) + s_B s_j abs(q)/pi arctan((1-psi)/(1+psi) tan theta/2)
$

This makes $psi$ the normalized minor radius. 
Sometimes, $zeta$ is called "toroidal" coordinate, while s is referred to as "poloidal" coordinate. 
However, this can be misleading. 
Varying $zeta$, while keeping $psi$ and $s$ constant will result in a screw like motion along both toroidally along $phi$ and poloidally along $theta$. // stimmt das überhaupt??
@fig:hamda:x visualizes how toroidal data is represented in hamada coordinates.

#include "../../figs/hamada/psi_const/fig.typ"

Both pictures in @fig:hamda:x do not represent the whole torus, but just its mantle.
The coordinate #sym.zeta has two discontinuities, as can be seen in @fig:hamada:x:t.
The toroidal discontinuity is at $phi = 0$, while the poloidal discontinuity is at $s=±0.5$ or $theta=±180°$ respectively.
This can be seen more clearly in the constant $phi$ case, which is visualized in @fig:hamda:phi.

Constant $phi$ visualization is of special interest in this thesis, as the purpose of ToPoVis and therefore this thesis is to create poloidal slices of the torus. // TODO: don't like this sentence

#include "../../figs/hamada/phi_const/fig.typ"

@fig:hamada:phi:t shows the poloidal slice in a polar coordinates with $s=text("const")$ lines added for reference.
The discontinuity of the #sym.zeta\-grid at $s=±0.5$ is visible clearly.
Also note that the s-grid is more dense on the left side (inner radius) than on the right side (outer radius).
This is caused by a discrepency between the regularly spaced s-grid and the non-uniform geometry of the torus. // passt das??

In hamada coordinates the poloidal slice is represented as a curved surface in 3d-space as can be seen in @fig:hamada:phi:h.
The surface can be described as a two dimensional function $zeta(psi,s)$, which is referred to as _#sym.zeta\-shift_ on the basis of #cite(<samaniego2024topovis>, form: "prose").
It spans across nearly the whole domain of the hamada coordinates.
One can recognise the unique shape of the curve in the #sym.zeta\-s-plane from @fig:hamda:x, which flattens out to a straight line when approaching $psi=0$.
@sec:background:topovis:zeta-shift discusses details on how #sym.zeta\-shift is defined and calculated using data from GKW.

=== Parallel Periodic Boundary Conditions
The s-grid is periodic across its boundary at $±0.5$ under the boundary condition // quelle !!

$ s = s±1 $

The regular spacing of the s-grid also extends across this periodic boundary condition.
To be precise the s-grid is defined as

#align(center)[
#table(
  columns: 2, align: horizon, stroke: none, gutter: 1em,
  [$ s_i = -0.5 + (Delta s)/2 + i dot Delta s$], [$ #text("with") 0 <= i <= N_s -1 $]
)]

The first value is $s_0 = -0.5 + (Delta s)/2$, while the last s-value will be denoted as $s_(-1) = 0.5 - (Delta s)/2$ for simplicity and in reference to advance array indexing.
This will create a gap between $s_0$ and $s_(-1)$ of exactly $Delta s$ across the periodic boundary.

A similar condition applies to #sym.zeta. If #sym.psi and s are held constant, #sym.zeta is perfectly periodic along the boundaries [0,1], meaing

$ zeta = zeta ± 1 $

Likewise, the #sym.zeta\-spacing is regular across its boundary, which leads to the discrete #sym.zeta\-grid being defined as follows.

#align(center)[
#table(
  columns: 3, align: horizon, stroke: none, gutter: 1em,
  [$ zeta_i = i dot Delta zeta$], [$Delta zeta = 1/N_zeta$], [$0 <= i <= N_zeta -1 $]
)]

The last #sym.zeta\-value is therefore $zeta_(-1) = 1-Delta zeta$.

However, if both #sym.zeta _and_ s are allowed to vary simultaneously other boundary conditions apply.

// TODO: Herleitung

// problem: regular grid interpolator can only interpolate

// s-grid is defined from $s=0.5-Delta s/2$ to $s=-0.5+Delta s/2$. Note how this gap increases with lower s-grid resolution.

// therefore we are left with a blank space without interpolated data between the first and last s.

// of course this gap could be filled otherwise, e.g. through linear interpolation. However, this would give a false confidence of how the potential looks like in that area.

// instead, when the flag '--periodic' is not supplied and the triangulation method is set to 'regular', ToPoVis will neither interpolate nor triangulate between $s=0.5-Delta s/2$ to $s=-0.5+Delta s/2$. This will lead to a white, blank space in that area.

// However, there is an option to generate additional gridpoints *outside* the domain. This is being done through _double-periodic boundary conditions_.

== Triangulation <sec:triang>

In the context of graph-theory or computational geometry _triangulation_ is the maximum set of edges for a given set of vertices so that no edges are intersecting @klein2005voronoi[p.233]. Given a set of points, there are many possible ways of performing a triangulation. However, not all triangles are created equal. // TODO: informal, is this okay?

#include "../../figs/triangulation/scattered/fig.typ"

@arbitrary_triangulation above illustrates, that arbitrary triangulations can lead to many acute triangles. This is undesirable in many cases, as it can lead to numerical artifacts @klein2005voronoi[p.234]. 
The triangles in @delaunay_triangulation on the other hand tend to be more equilateral leading to a more uniform representation of the grid @lucas2018delaunay. The _delaunay triangulation_ achieves this by maximizing the minimal interior angle of all triangles @klein2005voronoi[p.234]. // ??: add further information? e.g. the circumcircle criterion?

// But there are also limits to the delaunay triangulation, which comes from two assumptions. No subset of four points are on the same circumcircle and no subset of three points lie on a straight line @klein2005voronoi[p.234]. 

While delaunay triangulation works efficiently when the grid is more or less uniform, it doesn't perform well on non-uniform grids @lo2013multigrid[p.15]. // TODO: kinda misleading → cpu performance is not great under certain circumstances
Moreover, different examples from #cite(<lo2013multigrid>,form: "prose", supplement: [p.21]), #cite(<peethambaran2015delaunay>, form: "prose", supplement: [p.166ff]) and #cite(<liu2008delaunay>, form: "prose", supplement: [p.1269]) show, that delaunay triangulation on non-uniform grids leads to many acute or big triangles. This can be illustrized well by examining the delaunay triangulation results of a spiral distrubution.

// ??: are these images too small?
#include "../../figs/triangulation/spiral/fig.typ"

Even on a non-uniform grid like the spiral shown in @fig:spiral_linear, the delaunay triangulation results in a uniform representation of the grid. However, this triangulation is quiet sensitive to noise, as can be observed in @fig:spiral_noisy. This is caused by preffering tiny, but delaunay-conform triangles in high-density ares, instead of big but acute triangles in low-density areas. // TODO: what excactly is caused?
A possible refinement of the triangulation is to add more data points in areas with low density, as being shown in @fig:two_noisy_spirals #footnote[#cite(<ruppert1995delaunay>, form: "prose") and further #cite(<shewchuk1996triangle>, form: "prose") present more elaborate ways of refining a triangulation. Both offer algorithmical approaches to add the least amount of extra vertices to the grid so that no resulting angles exceed a given angle. _Triangle_ is an implementation of this and is freely availiable at https://www.cs.cmu.edu/~quake/triangle.html @shewchuck2025triangle.]. // ??: maybe move this to outlook? 

However, in the context of three-dimensional data, every point is assigned a value. In order to add more additional points interpolation is necessary. // for example with height data

== Interpolation
This section focuses on interpolation of multidimensional data, at this is the data being interpolated in ToPoVis is three-dimensional.
When faced with multidimensional data, there are multiple different interpolation approaches to choose from. 
In this section two different methods are being introduced and discussed: Firstly, the functionality of the `RegularGridInterpolator` will be discussed. The second section takes a look at the `RBFInterpolator` and points out the differences of the two.

=== RegularGridInterpolator
The `RegularGridInterpolator` (RGI) is a python class provided by the package `scipy.interpolate` @scipy2025rgi. 
The RGI is an interpolator, that is designed to work with N-dimensional data defined on a rectilinear grid; meaning a rectangular grid with even or uneven spacing @scipy2025rgi. 
Different methods of interpolation are supported, namely nearest-neighbor, linear, or spline interpolations @scipy2025rgi.
Higher degree spline interpolations usually yield more accurate results, at the cost of more expensive computations compared to linear interpolation @scipy2025rgi.
However, due to the assumption of a regular grid structure, the RGI avoids expensive triangulation and operates efficiently even when constructing splines of degree 3 or 5 @scipy2025rgi. 
On regular grids multivariate (or N-dimensional) linear interpolation can be done by interpolating consecutively in each dimension @weiser1988interpolation. 
When constructing splines, this involves solving a large sparse linear system @scipy2025rgi.

// TODO: Abschlusssatz

// Bild maybe ??

=== RBFInterpolator
A lot of data sets from real world applications are not defined on a regular grid. 
Examples of this include measurements from meteorological stations, which are scattered irregularly, images with corrupted pixels, as well as potential or vector data from physical experiments @skala2016rbf[p.6].
Many interpolation methods such as multivariate splines, Clough-Tocher or finite element interpolators expect a regularly structured grid, or depend on the prior creation of a mesh (usually through expensive triangulation) @wendland2004scattered[p.ix]. Radial basis functions (RBF) provide a truly meshless alternative for interpolation on scattered data @wendland2004scattered[p.ix]. 

The python package `scipy.interpolate` provides an implementation of a radial basis function interpolator, namely `RBFInterpolator` (RBFI) @scipy2025rbf.
Similar to the RGI, the RBFI is a class provided by  and also works on N-dimensional data @scipy2025rbf. 
A radial basis function (RBF) is a N-dimensional scalar function $f(r)$, that only depends on the distance $r=|x-c|$ between the evaluation point $x$ and the center $c$ of the RBF @scipy2025rbf.
There are two main kinds of RBFs commonly used, also referred to as _kernels_. Firstly, infinitely smooth functions like the gaussian function $f(r)=e^(-(#sym.epsilon r)^2)$, an inverse quadratic $f(r)=frac(1, 1+(#sym.epsilon r)^2)$ or inverse multiquadric function $f(r)=frac(1,sqrt(1+(#sym.epsilon r)^2))$ @scipy2025rbf. 
Note that these are scale variant and therefore require finding an appropriate smoothing parameter $#sym.epsilon$ (e.g. through cross validation, or machine learning) @scipy2025rbf.
Secondly, a polyharmonic spline of form $f(r)=r^k$ with uneven $k$ or $f(r)=r^k ln(r)$ with even $k$ can be used @scipy2025rbf. 
The default kernel used by the RBFI is also a polyharmonic spline called _thin plate spline_, which is defined as $f(r)=r^2 ln(r)$ @scipy2025rbf.
The interpolant takes the form of a linear combination of RBFs centered at each data point respectively, which can then be evaluated at arbitrary points @scipy2025rbf.
As memory required to solve the interpolation increases quadratically with the number of data points, the number of nearest neighbors to consider for each evaluation point can be specified @scipy2025rbf.

// this is similar to blurring a pixelated low resolution image to generate a blurry but smooth image, just that it works with unstructured grid
// maybe better example: corrupted grid data
// our visual perception works similarly btw

// ??: enumerate kernels in list?
// compare details in a table??

== ToPoVis
=== What is ToPoVis?
_ToPoVis_ is a python script developed by Sofia Samaniego in 2024 @samaniego2024topovis. It aims to compute and visualize poloidal cross sections of eletrostatic potential #sym.Phi inside a tokamak, hence the name "ToPoVis" (#strong[To]kamak #strong[Po]loidal cross section #strong[Vis]ualisation) @samaniego2024topovis[p.10,72]. ToPoVis works with the simulation data outputted by GKW.


// TODO: Output Bild von ToPoVis einfügen

// both linear and non-linear simulations
// geometries: circular, s-alpha, chease-global

=== How does ToPoVis work?

The program can be subdivided into the following sequences: //TODO: not happy with this

#set enum(numbering: "1.", full: true)
1. Reading values from `gkwdata.h5` file.
2. Calculating #sym.zeta for a given #sym.phi, also known as #sym.zeta\-shift
3. Evaluating potential #sym.Phi
  1. Linear case: Calculating potential #sym.Phi using fourier eigenmodes $hat(f)$
  2. Non-linear case: Interpolating 3D-potential and evaluating it at #sym.zeta\-shift
4. Plotting the potential on poloidal slice

// TODO: maybe add flow chart?

==== Calculating the #sym.zeta\-shift <sec:background:topovis:zeta-shift>
A poloidal slice implies satisfying the condition $#sym.phi = #text("const")$. The way the #sym.zeta\-shift is calculated is different for the kind of geometry being used for the simulation. ToPoVis works for three different geometries: 1) circular, 2) CHEASE and 3) s-#sym.alpha. In each geometry the transformations between toroidal and hamada coordinates are different @samaniego2024topovis[p.20ff].

In all geometries #sym.zeta is the only coordinate that is dependend on #sym.phi and can be defined generally as follows:

$ #sym.zeta = -frac(#sym.phi, 2#sym.pi) + G(psi, s) $

With each geometry having its own definition of $G$. // TODO: formulation 

===== Circular geometry
// #sym.psi\(r) &= frac(r, R_#text("ref")) \
// s(r, #sym.theta) &= frac(1, 2#sym.pi) [#sym.theta + #sym.psi\(r) sin(#sym.theta)] \
In circular geometry, the factor $G$ is defined like follows @peeters2015[p.25].
$ 
  G(psi, s) = frac(1,#sym.pi) s_B s_j abs(q(#sym.psi)) arctan[sqrt(frac(1-#sym.psi, 1+#sym.psi)) tan frac(#sym.theta, 2)] = #text("gmap")\(psi, s)
$ 
It is outputted directly by GKW and can be found under `geom/gmap` as a discrete function of #sym.psi and $s$ @samaniego2024topovis[p.21]. // TODO: don't call the discrete function

===== CHEASE geometry
For general geometries GKW interfaces the CHEASE code @peeters2015[p.2]. CHEASE (Cubic Hermite Element Axisymmetric Static Equilibrium) is a solver for toroidal magnetohydrodynamic equilibria developed by #cite(<lutjens1996chease>, form: "prose"). Unlike GKW, it treats the plasma as a fluid rather then a many-particle system. CHEASE can deal with general geometries, e.g. geometries with up-down-asymmetric cross sections as present in tokamaks like JET (Joint European Torus) and the planned ITER (International Thermonuclear Experimental Reactor) @lutjens1996chease[p.221f]. Hence, these general geometries are named CHEASE or CHEASE-global in GKW @peeters2015[p.42].
The resulting geometry factor can be expressed as @samaniego2024topovis[p.21]:

// !! citation other than topovis?

$ G(psi, s) = underbrace(s_B s_j frac(F(Psi) J_Psi _(zeta s)(Psi), 4pi^2), text("gmap")(psi, s)) integral_0^s d tilde(s) frac(1, R^2(psi, tilde(s))) $

The prefactor of the integral is calculated by GKW and is outputted to `geom/gmap` as a discrete 2D-function @samaniego2024topovis[p.21]. The $s$-integral, however, is being calculated by ToPoVis. This is being done using numerical trapezoidal integration @samaniego2024topovis[p.21]. In the usual case of an even number of grid points in the $s$ direction $s=0$ and therefore $R(s=0)$ doesn't not exist and is interpolated using B-splines @samaniego2024topovis[p.21]. 

// ?? weiter ausführen?

===== s-#sym.alpha geometry
The s-#sym.alpha geometry is the first order approximation of the circular geometry. Flux surfaces are circular and having a small inverse aspect ratio $psi = r/R << 1$ @peeters2015[p.23]. Because of the heavily simplified nature of this, it can lead to numerical instability for non-linear simulations @peeters2015[p.23]. In this geometry the geometry factor $G$ is defined like this @peeters2015[p.23] #footnote[In previous versions of @peeters2015 #sym.zeta was falsely defined as $zeta = frac(s_B s_j, 2pi) [abs(q) theta - phi]$.]:

$ G(psi, s) = s_B s_j abs(q(psi)) s $

This time, `geom/gmap` is not being used. Instead $G$ is calculated in ToPoVis through the geometry factors `input/geom/signB`, `input/geom/signJ` and `geom/q` exported by GKW.

==== Linear simulations
For linear simulations the calculation of the poloidal potential is simple, as a function in hamada coordinates $f(#sym.psi, #sym.zeta, s)$ can be expressed as a Fourier series.

$ sum_k_#sym.zeta hat(f)(#sym.psi, k_#sym.zeta, s) exp(i k_#sym.zeta #sym.zeta) $

$ f(#sym.psi, #sym.zeta, s) = hat(f)(#sym.psi, #sym.zeta, s) exp(i k_#sym.zeta #sym.zeta) + hat(f^*)(#sym.psi, #sym.zeta, s) exp(-i k_#sym.zeta #sym.zeta) $


==== Non-linear simulations


=== What needs to be improved?

// ! Numerical artifacts

// readability of the code

// vectorize / optimize some calculations


#load-bib()