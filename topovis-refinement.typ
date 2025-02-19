#import "bib.typ" : load-bib

= Refinement of the ToPoVis Code
// TODO: move this to introduction or first chapter
== What is GKW?
The _Gyrokinetic Workshop (GKW)_ is a code to simulate and study turbolences of a confined plasma, usually inside of a tokamak @peeters2015[p.1]. It is written in Fortran 95 and was initially developed at the University of Warwick in 2007 @peeters2015[p.1]. The code is freely availiable and currently being hosted at https://bitbucket.org/gkw/gkw. 

== What is ToPoVis
_ToPoVis_ is a python code developed by Sofia Samaniego in 2024 @samaniego2024topovis. It aims to compute and visualize poloidal cross sections of eletrostatic potential #sym.Phi inside a tokamak, hence the name "ToPoVis" (#strong[To]kamak #strong[Po]loidal cross section #strong[Vis]ualisation) @samaniego2024topovis[p.10,72]. ToPoVis works with the simulation data outputted by GKW. 

// TODO: Output Bild von ToPoVis einfügen

// one of main accomplishments: zeta-shift → brief explanation how that works

// TODO: do I even need this section?
== Former issues of ToPoVis 

// TODO: section name
== Interpolation
// one main issue: numerical artifacts caused by triangulation on non-uniform grid

// visualize this with an image

// solution: interpolation of said grid

// interpolation happens on regular hamada grid

// interpolation results: picture comparision

== Extrapolation through double-periodic boundary conditions


#load-bib()