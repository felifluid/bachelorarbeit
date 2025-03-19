#import "@preview/subpar:0.2.1"
#import "../functions.typ" : load-bib,  enum_numbering

= Background
== GKW
The _Gyrokinetic Workshop_ (GKW) is a code to simulate and study turbolences of a confined plasma, usually inside of a tokamak @peeters2015[p.1]. It is written in Fortran 95 and was initially developed at the University of Warwick in 2007 @peeters2015[p.1]. The code is freely availiable and currently being hosted at https://bitbucket.org/gkw/gkw. It works in both linear and non-linear regimes @peeters2015[p.1]

// linear case
// GKW solves eigenvalues k_zeta (fourier eigenmodes)
// eigenvalues are symmetrical
// as "linear" they do not couple and evolve independently
// usually only one fourier mode is simulated
// → fourier coefficient is 2D (psi, s) with k_zeta constant

// nonlinear run


== Hamada coordinates

// image of a tokamak torus with hamada coordinates

== Triangulation <sec:triang>

In the context of graph-theory or computational geometry _triangulation_ is the maximum set of edges for a given set of vertices so that no edges are intersecting @klein2005voronoi[p.233]. Given a set of points, there are many possible ways of performing a triangulation. However, not all triangles are created equal. // TODO: informal, is this okay?

#subpar.grid(
  figure(
    image("../../figs/random_triangles.svg"), 
    caption: [
      arbitrary
    ]
  ), <arbitrary_triangulation>,
  figure(
    image("../../figs/delaunay_triangles.svg"), 
    caption: [
      delaunay
    ]
  ), <delaunay_triangulation>,
  columns: (1fr, 1fr),
  caption: [
    Comparison of two triangulations of the same set of points
  ],
  label: <triangulation_comparison>,
)

@arbitrary_triangulation above illustrates, that arbitrary triangulations can lead to many acute triangles. This is undesirable in many cases, as it can lead to numerical artifacts @klein2005voronoi[p.234]. 
The triangles in @delaunay_triangulation on the other hand tend to be more equilateral leading to a more uniform representation of the grid @lucas2018delaunay. The _delaunay triangulation_ achieves this by maximizing the minimal interior angle of all triangles @klein2005voronoi[p.234]. // ??: add further information? e.g. the circumcircle criterion?

// But there are also limits to the delaunay triangulation, which comes from two assumptions. No subset of four points are on the same circumcircle and no subset of three points lie on a straight line @klein2005voronoi[p.234]. 

While delaunay triangulation works efficiently when the grid is more or less uniform, it doesn't perform well on non-uniform grids @lo2013multigrid[p.15]. // TODO: kinda misleading → cpu performance is not great under certain circumstances
Moreover, different examples from #cite(<lo2013multigrid>,form: "prose", supplement: [p.21]), #cite(<peethambaran2015delaunay>, form: "prose", supplement: [p.166ff]) and #cite(<liu2008delaunay>, form: "prose", supplement: [p.1269]) show, that delaunay triangulation on non-uniform grids leads to many acute or big triangles. This can be illustrized well by examining the delaunay triangulation results of a spiral distrubution.

// ??: are these images too small?
#subpar.grid(
  figure(
    image("../../figs/triangulation/spiral_lin.svg"),
    caption: [spiral]
  ), <fig:spiral_linear>,
  figure(
    image("../../figs/triangulation/spiral_noisy.svg"),
    caption: [noisy spiral],
  ), <fig:spiral_noisy>,
  figure(
    image("../../figs/triangulation/spiral_interpolation.svg"),
    caption: [two noisy spirals],
  ), <fig:two_noisy_spirals>,
  columns: (1fr, 1fr, 1fr),
  caption: [
    Comparison of delaunay triangulations for three spiral distrubutions.
  ],
  label: <fig:delaunay_spiral>,
)

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
Similar to the RGI, the `RBFInterpolator` (RBFI) is a class provided by `scipy.interpolate` and also works on N-dimensional data @scipy2025rbf. 
Unlike the RGI, which only operates with data on a regular grid, the RBFI also works with unstructured data @scipy2025rbf.
A radial basis function (RBF) is a N-dimensional scalar function $f(r)$, that only depends on the distance $r=|x-c|$ between the evaluation point $x$ and the center $c$ of the RBF @scipy2025rbf.
There are two main kinds of RBFs commonly used, also referred to as _kernels_. Firstly, infinitely smooth functions like the gaussian function $f(r)=e^(-(#sym.epsilon r)^2)$, an inverse quadratic $f(r)=frac(1, 1+(#sym.epsilon r)^2)$ or inverse multiquadric function $f(r)=frac(1,sqrt(1+(#sym.epsilon r)^2))$ @scipy2025rbf. 
Note that these are scale variant and therefore require finding an appropriate smoothing parameter $#sym.epsilon$ (e.g. through cross validation, or machine learning) @scipy2025rbf.
Secondly, a polyharmonic spline of form $f(r)=r^k$ with uneven $k$ or $f(r)=r^k ln(r)$ with even $k$ can be used @scipy2025rbf. 
The default kernel used by the RBFI is also a polyharmonic spline called _thin plate spline_, which is defined as $f(r)=r^2 ln(r)$ @scipy2025rbf.
The interpolant takes the form of a linear combination of RBFs centered at each data point respectively, which can then be evaluated at arbitrary points @scipy2025rbf.
As memory required to solve the interpolation increases quadratically with the number of data points, the number of nearest neighbors to consider for each evaluation point can be specified @scipy2025rbf.

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

==== Calculating the #sym.zeta\-shift
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