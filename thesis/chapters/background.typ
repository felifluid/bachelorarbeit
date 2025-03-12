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
// focus on multidimensional data, at this is the data dimensions being interpolated in topovis
When faced with multidimensional data, there are multiple different interpolation approaches to choose from. In this section two different methods are being introduced and discussed.

=== RegularGridInterpolator
The `RegularGridInterpolator` (RGI) is a python class provided by SciPy @scipy2025rgi. This class makes a few assumptions of the grid structure to avoid expensive triangulation and therefore speed up the interpolation @scipy2025rgi. That is, the grid must be rectilinear, means rectangular with even or uneven spacing @scipy2025rgi. The RGI supports different methods for interpolation, among them nearest, linear, cubic and quintic @scipy2025rgi. The last two involve solving a large sparse linear system @scipy2025rgi.

// TODO: explain trilinear interpolation?

// ??: different headline?
=== RBFInterpolator
Similar to the RGI The `RBFInterpolator` (RBFI) is also a class provided by SciPy @scipy2025rbf. 

// TODO: cite underlying papers

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
  1. Linear case: Calculating potential #sym.Phi using fourier eigenmodes $hat(f)$
  2. Non-linear case: Interpolating 3D-potential and evaluating it at #sym.zeta\-shift
3. Plotting the potential on poloidal slice

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