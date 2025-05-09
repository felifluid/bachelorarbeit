#import "@preview/subpar:0.2.1"
#import "@preview/ouset:0.2.0": ouset
#import "../functions.typ" : enum_numbering

= Background <chap:background>
#include "../../figs/tokamak/fig.typ"
== A Word to Nuclear Fusion
In short, nuclear fusion is the process of two nuclei fusing together to form a heavier nucleus, which results in a convertion of mass to energy given by Einstein's mass-energy formula.
This usually involves increasing the temperature of an ionized gas called plasma to around 100 million degrees.
The challenge is then to confine this high temperature plasma, which is commonly done by strong magnetic fields with a device called _tokamak_ (see @fig:tokamak) @li2014tokamak[p.1].

The current of the moving plasma itself creates a poloidal magnetic field.
Additionally, both poloidal and toroidal external magnetic field are applied simultaniously to confine and shape the plasma.
Altogether this results in a magnetic field that twists helically around the torus @samaniego2024topovis[p.11].

== GKW
The _Gyrokinetic Workshop_ (GKW) is a code written in Fortran 95 to simulate and study turbolences of a confined plasma, usually inside of a tokamak. 
This involves solving the five-dimensional gyrokinetic equation, hence the name.
Due to the computational complexity it uses a highly scalable parallel approach.
GKW supports both linear and non-linear regimes, for both local and global toroidal simulations, while this thesis will only focus on linear and non-linear global runs for all purposes.
The code is freely availiable and currently being hosted at https://bitbucket.org/gkw/gkw. 
For indepth information about GKW and how it works see #cite(<peeters2015>, form: "prose").

== Hamada Coordinates
To efficiently solve the gyrokinetic equation, GKW makes use of so called _Hamada Coordinates_.
Before that, a quick view into toroidal coordinates is needed.
@fig:cylindrical-toroidal shows the relation between cylindrical $(R,Z,phi)$ and toroidal $(psi, theta, phi)$ coordinates. 
In it, the helical magnetic field lines for $psi="const"$ are also shown.

#include "../../figs/cylindrical_coords/fig.typ"

The relation can be expressed via the following equations @samaniego2024topovis[p.11]:

$
  &R = R_0 + psi cos theta \
  &Z = psi sin theta \
  &phi = phi
$ <eq:cylindrical_toroidal>

Hamada coordinates are retrieved by transforming the toroidal $(psi,theta,phi)$ coordinates in such a way that

+ field lines become straight and
+ one of the coordinates is aligned with the magnetic field.

This leads to the following generalized equations

$
  &psi(r) = r/R_"ref" #<eq:psi> \
  &s(theta, psi) = integral_0^theta ("d"theta')/(bold(B) dot nabla theta') slash.big integral.cont ("d"theta')/(bold(B) dot nabla theta') #<eq:s> \
  &zeta(psi, theta) = q s - phi/(2pi) - g(theta, psi) #<eq:zeta>
$ <eq:hamada>

For further reading on how this is achieved in detail, see @peeters2015[20ff] and @peeters2015[Appen. A].
Sometimes, $zeta$ is called the _toroidal_ coordinate, while s is referred to as the _poloidal_ coordinate. 
However, this can be misleading as varying $zeta$ at lconstant $psi$ and $s$, will result in a screw like motion along both toroidally along $phi$ and poloidally along $theta$.
The so called _safety factor_

$ q(psi) = B^gamma/B^s $

describes how many toroidal turns a magnetic field line takes before returning to its starting poloidal angle.
A safety factor of $q=5/4$ would mean, the magnetic field line reaches its starting position after excactly one and a fourth toroidal turn.
Generally, higher values of $q$ result in better plasma stability.
In GKW the sign of the safety factor is oftentimes normalized to 

$ q = s_B s_j abs(q) $ <eq:q>

where $s_B = ±1$ and $s_j = ±1$ represent the sign of the magnetic field and the plasma current @peeters2015[p.21].

Using above transformation the coordinates are normalized and made unitless. 
The radial coordinate $psi$ is retrieved by normalizing the minor radius $r$ to the major radius $R_text("ref")$, while the $s$-grid is normalized to the interval $[-0.5, 0.5]$.

Note that the generalized coordinates in Equations @eq:hamada[] just assume a toroidal geometry, while the exact shape of the tokamak geometry is left uncertain.
A few different geometries are implemented, namely 1) circular, 2) CHEASE as well as 3) the simplified circular s-#sym.alpha geometry.
The next section takes a look at Hamada Coordinates in _circular_ coordinates.

=== Circular Case
In circular geometry the coordinate transformation is defined by the following equations (see @peeters2015[A1])

$
  &psi(r) = r/R_text("ref") \ 
  &s(r, theta) = 1/(2 pi) (theta + psi sin theta) \
  &zeta (r, theta, phi) = - phi/(2 pi) + s_B s_j abs(q)/pi arctan((1-psi)/(1+psi) tan theta/2)
$

Assuming $s_B s_j = +1$ the $zeta$-grid can range from 

$ zeta_text("min") = -1-abs(q)/2 $ 

for $phi=2pi$ to 

$ zeta_text("max") = abs(q)/2 $ 

in the case of $phi=0$. 
The $zeta$-grid is oftentimes normalized to $[0,1]$, however this mapping is not generally continuous, as will be explained in @sec:background:hamada:periodicity.

For easier understanding of how toroidal and hamadian coordinates are transformed, Figure @fig:hamda:x[] & @fig:hamda:phi[] show side by side comparisons of $psi="const"$ and $phi="const"$ surfaces respectively.

#include "../../figs/hamada/psi_const/fig.typ"

@fig:hamada:x:t shows the torus in cartesian coordinates.
Both plots do not represent the whole torus, but just its outest mantle ($psi=psi_"max"$).
The grid in @fig:hamada:x:h has been adjusted, to emphasize the relationship of $zeta$ and $q$.
The top line is $zeta(s, phi=0)$, while the bottom line represents $zeta(s, phi=2pi)$ with all other toroidal angles lying in between. // TODO: umformulieren
This leads to a constant width of $Delta zeta=1$.
Mapped to a torus $zeta$ has two discontinuities, as can be seen in @fig:hamada:x:t.
The toroidal discontinuity is at $phi = 0$, while the poloidal discontinuity is positioned at $s=±0.5$ or $theta=±pi$ respectively.
This can be seen more clearly in the constant $phi$ case, which is visualized in @fig:hamda:phi.

#include "../../figs/hamada/phi_const/fig.typ"

@fig:hamada:phi:t shows the poloidal slice in polar coordinates with $s=text("const")$ and $psi="const"$ lines added for reference.
The discontinuity of the $zeta$\-grid at $s=±0.5$ is visible clearly.
Also note that the s-grid is more dense on the left side (inner radius) than on the right side (outer radius).
This is caused by a discrepency between the regularly spaced s-grid and the non-uniform geometry of the torus. // passt das??

In hamada coordinates the poloidal slice is represented as a curved surface in 3d-space as can be seen in @fig:hamada:phi:h.
The surface can be described as a two dimensional function $zeta(psi,s)$, which is referred to as _$zeta$-shift_ on the basis of #cite(<samaniego2024topovis>, form: "prose").
One can recognise the unique shape of the curve in the $zeta$-$s$-plane from @fig:hamada:x:h, which flattens out to a straight line when approaching $psi=0$.
@sec:background:topovis:zeta-shift discusses details on how $zeta$-shift is defined and calculated using data from GKW.

=== Periodicity <sec:background:hamada:periodicity>
==== Toroidal Periodicity
Trivially the toroidal angle $phi$ is periodic:

$ phi = phi ± 2pi $ <eq:periodic:phi>

We can generalize @eq:zeta further to get

$ zeta = -phi/(2pi) + G(psi, s) $ <eq:zeta_general>

Equations @eq:periodic:phi[] and @eq:zeta_general[] can be combined to get the periodic boundary condition

$ zeta = zeta ± 1 $ <eq:periodic:zeta>

as both $psi$ and $s$ are no functions of $phi$. 
This again can be generalized for any function

$ f(psi, zeta, s) = f(psi, zeta ± 1, s) $ <eq:toroidal_periodicity>

The spectral representation stays unaffected by this @peeters2015[p.44].

==== Poloidal Periodicity
For the torus poloidal periodicity is simply expressed as

$ theta = theta ± 2pi $ <eq:periodic:theta>

If both $psi$ and $zeta$ are held constant, this periodicity translates directly to 

$ s = s±1 $ <eq:periodic:s>

But since in hamada coordinates $s(theta, psi)$ and therefore also $zeta(phi, psi, s(theta, psi))$ are functions of $theta$, in general something called _double periodicity_ or _parallel periodicity_ occurs.

The quantity $G(psi, s)$ in #ref(<eq:zeta_general>) is not periodic in general. 
However, we can combine this with @eq:zeta to find

$ G(psi, s, theta) = q(psi) s - g(psi, theta) $

Note, that this is now a function of $theta$ instead of $s$. 
However, at constant $psi$, every poloidal position is uniquely defined by either its angle $theta$ or its $s$ coordinate.
Hence, $g(psi,theta)$ can be expressed as $g(psi, s)$ for constant $psi$.

One can also find, that for $psi="const"$ the function $g(psi, s)$ must be periodic in $s$.
When following a field line $zeta$ one poloidal turn at a constant $psi$ from $theta_A = 0$ to $theta_B = ±2pi$, this corresponds to a change in $s$ from $s_A = 0$ to $s_B = ±1$.
Due to poloidal periodicity, point $A$ and $B$ must be at the same poloidal position, meaning

$ Delta theta = theta_B - theta_A = 0 $

and therefore

$ 
  &g(psi, theta) = g(psi, theta±2pi)  #<eq:periodic:g:theta> \
  &g(psi,s) = g(psi, s±1) #<eq:periodic_g>
$

Hence, and the quantity $G(psi, theta)$ can be written as

$ G(psi, s) = q s - g(psi, s) $ <eq:big_G>

with $g(psi,s)$ being periodic in $s$.

Again, consider following a field line one poloidal turn and evaluate

$
  G(psi, s±1) &ouset(=, #ref(<eq:big_G>,supplement: [])) q (s±1) - g(psi, s±1) \
              &ouset(=, #ref(<eq:periodic_g>, supplement: [])) q s ± q - g(psi, s) \
              &ouset(=, #ref(<eq:big_G>,supplement: [])) G(psi, s) ± q
$ <eq:periodic:G>

With this a function for $zeta$ can be derived like

$ 
  zeta(psi, s±1) &ouset(=, #ref(<eq:zeta_general>, supplement: [])) -phi/(2pi) + G(psi, s±1) \
                 &ouset(=, #ref(<eq:periodic:G>, supplement: [])) -phi/(2pi) + G(psi, s) ± q \
                 &ouset(=, #ref(<eq:zeta_general>, supplement: [])) zeta(psi, s) ± q
$ <eq:double_periodic_zeta>

Poloidal double periodicity can be generalized for any given function $f(psi, zeta, s)$

// TODO: add box
$ f(psi, zeta, s) = f(psi, zeta ∓ q, s±1) $ <eq:double_periodicity>

In the case of a function $f(psi, s)$ that is not depended on $zeta$, the above equation simplifies to

$ f(psi, s) = f(psi, s±1) $

while toroidal periodicity (see @eq:toroidal_periodicity) is trivially satisfied.

Considering the spectral representation

$ f(psi, zeta, s) = sum_(k_zeta) hat(f)(psi,k_zeta,s) exp(i k_zeta zeta) $

the poloidal double periodic boundary condition translates to

// TODO: add box
$ f(psi,k_zeta,s) = hat(f)(psi,k_zeta,s±1) exp(∓i k_zeta q) $ <eq:spectral_double_periodicity>

=== Specifications of the discrete grid <sec:discrete_grid>
As GKW solves the gyrokinetic equation numerically, it does so on a discrete hamada grid

$ 
  H = {(psi_i, s_j, zeta_k) in RR^3 | i in [0, N_psi -1] , j in [0, N_s -1], k in [0, N_zeta -1] } 
$ <eq:discrete_hamada>

with $i,j,k in NN$.

The grid is equally spaced, specified by the following equations. The implementation of this can be found in the GKW code in the module `geom.f90` in the subroutine `geom_init_grids` @peeters2015code. 

The radial component $psi_i$ is defined as

$ psi_i = psi_l + i dot Delta psi #h(2cm) Delta psi = (psi_h - psi_l)/(N_psi)  $ 

starting at the center $psi_l$ extending outwards to $psi_h$. 

The $zeta_k$-grid is defined as

$
  zeta_k = k dot Delta zeta \
  Delta zeta = L_zeta/N_zeta 
$

with // ?? what is this?

$ L_zeta = 2pi rho_* / k_(zeta, "min") = 1/n_"min" $

using

$ k_(zeta,"min") = 2pi n_"min" rho_* $ // das überhaupt erwähnen??

where $rho_*$ is the normalized Larmor radius.
In this $n_"min"$ denotes the the fraction of the torus data that was simulated @samaniego2024topovis[p.19].
Therefore the discrete $zeta$-grid is defined between $zeta_0 = 0$ and 
$zeta_(-1) = L_zeta - Delta zeta$.
This means, that even if $L_zeta=1$ the grid will have a spacing of $Delta zeta$ between the first and last grid point accounting for the periodic boundary condition $zeta = zeta±1$.

Similarly such a gap is also present for the $s$-grid. The grid is defined as

$ s_j = s_0 + j dot Delta s $ <eq:discrete_s>

with

$
  &s_0 = -0.5 + (Delta s)/2 \
  &Delta s = 1/N_s
$

which makes $s_(-1) = 0.5 - (Delta s)/2$ the maximum $s$ value.
This again leaves a gap of $Delta s$ across the periodic boundary because of $s=s±1$. 

// Will ich diesesn Teil hier drin haben als Teaser??
Normally, this wouldn't be a problem. // TODO: ehh 🤷
However, in the context of interpolation these gaps lead to complications.
Using conventional methods, interpolation is only possible between the first the last grid point. 
This would leave a blank gap with the width of the grid-spacing, where no data can be generated.
To circumvent this, the grid has to be extended using periodic boundary conditions.
Details on how this is dealt with will be discussed in @sec:topovis:interpolation.

== Triangulation <sec:triang>

In the context of graph-theory or computational geometry _triangulation_ is the maximum set of edges for a given set of vertices so that no edges are intersecting @klein2005voronoi[p.233]. Given a set of points, there are many possible ways of performing a triangulation. However, not all triangles are created equal. // TODO: informal, is this okay?

#include "../../figs/triangulation/scattered/fig.typ"

@fig:triangulation:scattered above shows two different triangulations of the same set of points. 
@fig:triangulation:scattered:arbitrary illustrates, that arbitrary triangulations can lead to many acute triangles. 
This is undesirable in many cases, as it can lead to numerical artifacts @klein2005voronoi[p.234]. 
The triangles in @fig:triangulation:scattered:delaunay on the other hand tend to be more equilateral leading to a more uniform representation of the grid @lucas2018delaunay.
The _delaunay triangulation_ achieves this by maximizing the minimal interior angle of all triangles @klein2005voronoi[p.234]. // ??: add further information? e.g. the circumcircle criterion?

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

=== RegularGridInterpolator <sec:background:rgi>
The `RegularGridInterpolator` (RGI) is a python class provided by the package `scipy.interpolate` @scipy2025rgi. 
The RGI is an interpolator, that is designed to work with N-dimensional data defined on a rectilinear grid; meaning a rectangular grid with even or uneven spacing @scipy2025rgi. 
Different methods of interpolation are supported, namely nearest-neighbor, linear, or spline interpolations @scipy2025rgi.
Higher degree spline interpolations usually yield more accurate results, at the cost of more expensive computations compared to linear interpolation @scipy2025rgi.
However, due to the assumption of a regular grid structure, the RGI avoids expensive triangulation and operates efficiently even when constructing splines of degree 3 or 5 @scipy2025rgi. 
On regular grids multivariate (or N-dimensional) linear interpolation can be done by interpolating consecutively in each dimension @weiser1988interpolation. 
When constructing splines, this involves solving a large sparse linear system @scipy2025rgi.

// TODO: Abschlusssatz

// Bild maybe ??

=== RBFInterpolator <sec:background:rbfi>
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
_ToPoVis_ is a python script developed by Sofia Samaniego in 2024 @samaniego2024topovis. It aims to compute and visualize poloidal cross sections of eletrostatic potential $Phi$ inside a tokamak, hence the name "ToPoVis" (#strong[To]kamak #strong[Po]loidal cross section #strong[Vis]ualisation) @samaniego2024topovis[p.10,72]. ToPoVis works with the simulation data outputted by GKW.

// TODO: Output Bild von ToPoVis einfügen

// both linear and non-linear simulations
// geometries: circular, s-alpha, chease-global

The program can be subdivided into the following sequences: //TODO: not happy with this

#set enum(numbering: "1.", full: true)
1. Retrieving data from `gkwdata.h5` file.
2. Calculating $zeta$ for a given $phi$, also known as $zeta$-shift
3. Evaluating potential $Phi$
  1. Linear case: Calculating potential $Phi$ using fourier eigenmodes $hat(f)$
  2. Non-linear case: Interpolating 3D-potential and evaluating it at $zeta$-shift
4. Plotting the potential on poloidal slice

// TODO: maybe add flow chart?aims to 

=== Calculating the $zeta$\-shift <sec:background:topovis:zeta-shift>
A poloidal slice implies satisfying the condition $#sym.phi = #text("const")$. The way the $zeta$\-shift is calculated is different for the kind of geometry being used for the simulation. ToPoVis works for three different geometries: 1) circular, 2) CHEASE and 3) s-#sym.alpha. In each geometry the transformations between toroidal and hamada coordinates are different @samaniego2024topovis[p.20ff].

In all geometries $zeta$ is the only coordinate that is dependend on #sym.phi and can be defined generally as follows:

$ zeta = -frac(#sym.phi, 2#sym.pi) + G(psi, s) $ <eq:zeta_s>

With each geometry having its own definition of $G$. // TODO: formulation 

==== Circular geometry
// #sym.psi\(r) &= frac(r, R_#text("ref")) \
// s(r, #sym.theta) &= frac(1, 2#sym.pi) [#sym.theta + #sym.psi\(r) sin(#sym.theta)] \
In circular geometry, the factor $G$ is defined like follows @peeters2015[p.25].

$ 
  G(psi, s) = frac(1,#sym.pi) s_B s_j abs(q(#sym.psi)) arctan[sqrt(frac(1-#sym.psi, 1+#sym.psi)) tan frac(#sym.theta, 2)] = #text("gmap")\(psi, s)
$ 

It is outputted directly by GKW and can be found under `geom/gmap` @samaniego2024topovis[p.21].

==== CHEASE geometry
For general geometries GKW interfaces the CHEASE code @peeters2015[p.2]. CHEASE (Cubic Hermite Element Axisymmetric Static Equilibrium) is a solver for toroidal magnetohydrodynamic equilibria developed by #cite(<lutjens1996chease>, form: "prose"). Unlike GKW, it treats the plasma as a fluid rather then a many-particle system. CHEASE can deal with general geometries, e.g. geometries with up-down-asymmetric cross sections as present in tokamaks like JET (Joint European Torus) and the planned ITER (International Thermonuclear Experimental Reactor) @lutjens1996chease[p.221f]. Hence, these general geometries are named CHEASE or CHEASE-global in GKW @peeters2015[p.42].
The resulting geometry factor can be expressed as @samaniego2024topovis[p.21]:

// !! citation other than topovis?

$ G(psi, s) = underbrace(s_B s_j frac(F(Psi) J_Psi _(zeta s)(Psi), 4pi^2), text("gmap")(psi, s)) integral_0^s d tilde(s) frac(1, R^2(psi, tilde(s))) $ <eq:G:chease>

The prefactor of the integral is calculated by GKW and is outputted to `geom/gmap` as a discrete 2D-function @samaniego2024topovis[p.21]. The $s$-integral, however, is being calculated by ToPoVis. This is being done using numerical trapezoidal integration @samaniego2024topovis[p.21]. In the usual case of an even number of grid points in the $s$ direction $s=0$ and therefore $R(s=0)$ doesn't not exist and is interpolated using B-splines @samaniego2024topovis[p.21]. 

// ?? weiter ausführen?

==== s-#sym.alpha geometry
The s-#sym.alpha geometry is the first order approximation of the circular geometry. Flux surfaces are circular and having a small inverse aspect ratio $psi = r/R << 1$ @peeters2015[p.23]. Because of the heavily simplified nature of this, it can lead to numerical instability for non-linear simulations @peeters2015[p.23]. In this geometry the geometry factor $G$ is defined like this @peeters2015[p.23] #footnote[In previous versions of @peeters2015 $zeta$ was falsely defined as $zeta = frac(s_B s_j, 2pi) [abs(q) theta - phi]$.]:

$ G(psi, s) = s_B s_j abs(q(psi)) s $

This time, `geom/gmap` is not being used. Instead quantity $G$ is calculated in ToPoVis.

=== Calculating the potential
==== Linear simulations <sec:topovis:linear>
In linear simulations GKW represents all pertubed quantities as a Fourier series @peeters2015[p.43]

$ sum_k_zeta hat(f)(psi, k_zeta, s) exp(i k_zeta zeta) $ <eq:fourier_series>

In linear simulations, usually only one Fourier mode is simulated.
Because of this, only linear simulations with `nmod=1` are supported @samaniego2024topovis[p.27].
Therefore, @eq:fourier_series simplifies to

$ f(psi, zeta, s) = hat(f)(psi, zeta, s) exp(i k_zeta zeta) + hat(f)^*(psi, zeta, s) exp(-i k_zeta zeta) $

The complex fourier coefficients are provided in the dataset `diagnostic/diagnos_mode_struct/parallel`, called `parallel.dat` in short.
In it, several diagnostics connected with the parallel mode structure are stored.
These include the fourier coefficients of the pertubed density, the pertubed parallel energy or the pertubed potential.
A complete list and order of all diagnostics in `parallel.dat` can be found in #cite(<peeters2015>, form: "prose", supplement: "p.144").
All quantities are saved sequentially in a 1d-array of length $N_"mod" N_x N_s N_"sp"$.
To extract the fourier coefficients of the potential, `parallel.dat` is reshaped and accessed at the specific column - once for the real part of the fourier coefficient, and again for the imaginary part.
ToPoVis achieves this using the included methods `reshape_parallel_dat_multi`, `fouriercoeff_imag` and `fouriercoeff_real`.
Both parts are then combined to retrieve the complex fourier coefficient. Note that because $N_"mod"=1$, the fourier coefficients are a function of $psi$ and $s$ and no function of $zeta$.

Using this, the potential at the poloidal cross section is calculated using the formula

$ Phi(psi, s, zeta_s) = RR[hat(f)(psi, s) exp(i k_zeta zeta_s) + hat(f)^*(psi,s) exp(-i k_zeta zeta_s)] $


==== Non-linear simulations <sec:topovis:nonlin>
Unlike the linear case, in non-linear simulations the potential #sym.Phi is calculated by GKW during runtime and exported as such. // is this done always??
This is done for each timestep and saved as seperate datasets under `diagnostic/diagnos_fields/`.
Each dataset is referenced by the key `Poten` followed by an 8 digit long number, e.g. `Poten00000343`. 
If not specified otherwise, the last availiable dataset will be used, as it usually gives the most accurate results @samaniego2024topovis[p.77].
As multiple $zeta$-modes are simulated in non-linear simulations, the potential $Phi(s, psi, zeta)$ is now three-dimensional.

Oftentimes its more useful to look at the non-zonal potential, as it reveals more about the turbulence structures. // TODO: Quelle?
This is achieved by substracting the mean $mu_j$ over $s$ and $zeta$ for each $psi_j$ from the potential $Phi_(i j k)$ corresponding to $psi_j$, i.e.

$ 
  Phi'_(i j k) = Phi_(i j k) - mu_j
$
with
$
  mu_j = 1/(N_s N_zeta) sum_(i=0)^(N_s-1) sum_(k=0)^(N_zeta -1) Phi_(i j k)
$

To retrieve potential data of a poloidal slice, i.e. $phi = "const"$, the 3d-potential $Phi(s, psi, zeta)$ now needs to be evaluated at $zeta$-shift.
But since, $zeta$-shift does not generally coincide with the discrete $zeta$-grid, interpolation is needed @samaniego2024topovis[p.23].
ToPoVis supplies two different interpolation methods: B-Splines and interpolation via Fast Fourier Transformation.
Either one can be choosen interactively, when running ToPoVis on non-linear simulation data.
However, B-Spline interpolation is the preffered option, as it yielded better results in benchmarking @samaniego2024topovis[p.29ff].

=== Plotting and Data Export <sec:topovis:plotting>
Finally, after the potential $Phi(psi,s)$ is calculated, a heatmap plot is created using the method `tricontourf`  from `matplotlib.pyplot` @samaniego2024topovis[p.27].
The poloidal coordinates $R(psi, s)$ and $Z(psi, s)$ (see !!) needed for this are included in the dataset `geom`. // add reference to cylindrical → toroidal coords
A colorbar and outlines at $psi_"min"$ and $psi_"max"$ are added.
The area in the center, where no data was simulated, is filled white to hide inapplicable delaunay triangles.

Additionally ToPoVis writes the plotted data $Phi$, $R$ and $Z$ to a hdf5-file @samaniego2024topovis[p.28].