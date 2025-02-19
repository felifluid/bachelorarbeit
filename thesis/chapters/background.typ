#import "../functions.typ" : load-bib

= Background
== GKW
The _Gyrokinetic Workshop (GKW)_ is a code to simulate and study turbolences of a confined plasma, usually inside of a tokamak @peeters2015[p.1]. It is written in Fortran 95 and was initially developed at the University of Warwick in 2007 @peeters2015[p.1]. The code is freely availiable and currently being hosted at https://bitbucket.org/gkw/gkw. 

== Hamada coordinates

// image of a tokamak torus with hamada coordinates

== Triangulation

// Delaunay triangulation: what is is and how does it work

// potential flaws

== Interpolation

== ToPoVis
=== What is ToPoVis?
_ToPoVis_ is a python script developed by Sofia Samaniego in 2024 @samaniego2024topovis. It aims to compute and visualize poloidal cross sections of eletrostatic potential #sym.Phi inside a tokamak, hence the name "ToPoVis" (#strong[To]kamak #strong[Po]loidal cross section #strong[Vis]ualisation) @samaniego2024topovis[p.10,72]. ToPoVis works with the simulation data outputted by GKW.

// TODO: Output Bild von ToPoVis einfügen

// both linear and non-linear simulations
// geometries: circular, s-alpha, chease-global

=== How does ToPoVis work?

// zeta shift - how does it work

==== Linear simulations

==== Non-linear simulations


== What needs to be improved?

#load-bib()