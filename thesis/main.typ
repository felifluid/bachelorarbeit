#import "template.typ": apply-template
#import "@preview/numbly:0.1.0": numbly

#show: apply-template(
  title: "Improved Visualization of Turbulences in a Tokamak using refined Triangulation and Interpolation",
  faculty: "Faculty of Theoretical Plasma Physics (TPV)",
  author: "Feli Nara Celeste Berner",
  supervisors: ("Prof. Dr. Arthur Peeters", "Dr. Florian Rath"),
  degree: "Bachelor of Science (LA Gym)",
  city: "Bayreuth",
  logo: "../figs/logo.svg",
  submissionDate: datetime.today(),
  prefaceEN: [
    ToPoVis is a code that computes and visualizes small scale plasma turbulences in a tokamak.
    This thesis tackles the issue of numerical artifacts through interpolation and refined triangulation. 
    To implement these extensive additions, the ToPoVis code is rewritten.
    The accuracy of the interpolation is measured for both linear and non-linear simulations.
    The new code is publicly available and can be used and extended in future research.
  ],
  prefaceDE: [
    ToPoVis ist ein Programm zur Berechnung und Visualisierung klein-struktureller Plasmaturbulenzen in einem Tokamak.
    In dieser Arbeit wird das Problem der numerischen Artefakte durch Interpolation und verfeinerter Triangulation angegangen. 
    Um diese umfangreichen Ergänzungen zu implementieren, wird der ToPoVis-Code neu geschrieben.
    Die Genauigkeit der Interpolation wird sowohl für lineare als auch für nicht-lineare Simulationen gemessen.
    Der neue Code ist öffentlich zugänglich und kann in der zukünftigen Forschung verwendet und erweitert werden.
  ],
  acknowledgements: [
    I would like to express my gratitude to Dr. Florian Rath for his generous time and support.
    His infectious excitement as well as his patience and openness for answering questions is what convinced me of writing this thesis in the first place and what helped me get through it.

    I want to extend my thanks to Prof. Dr. Arthur for offering the opportunity of this thesis and taking his time for our weekly meetings.

    I would further like to thank everyone who has supported me during the time of this research:
    Leonard Schmid for enduring my infodumps and providing new valuable perspectives.
    Sara Kleyman for her encouragements and being my body double numerous times.
    Björn Luchterhandt for inspiring me to learn Typst and sharing his knowledge about it.
    Elli Quitter for always caring for and comforting me in times of need.

    Thanks to all of you and everyone who showed interest in this work.
  ],
  disclaimerSignature: image(width: 6cm, "../figs/signature.jpg")
)[

#include "chapters/introduction.typ"

#include "chapters/background.typ"

#include "chapters/topovis-refinement.typ"

#include "chapters/conclusion.typ"

#include "chapters/outlook.typ"

// Addendum
#set heading(
  numbering: numbly(
    "{1:A.}",
    "{2:1.}",
    "{2:1}.{3:1}.",
    "{2:1}.{3:1}.{4:1}.",
))

#set heading(supplement: [Appendix])
#counter(heading).update(0)

#include "chapters/usage.typ"

]