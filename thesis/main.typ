#import "template.typ": apply-template
#import "@preview/numbly:0.1.0": numbly

#show: apply-template(
  title: "Improved Visualization of Turbulences in a Tokamak using refined Triangulation and Interpolation",
  faculty: "Faculty of Theoretical Plasma Physics \ TPV",
  author: "Feli Nara Celeste Berner",
  supervisors: ("Prof. Dr. Arthur Peeters", "Dr. Florian Rath"),
  degree: "Bachelor of Science (LA Gym)",
  city: "Bayreuth",
  logo: "../figs/logo.svg",
  submissionDate: datetime.today(),
  preface: [
    ToPoVis is a code that computes and visualizes plasma structures as poloidal cross sections in global tokamak geometry.
    This thesis tackles the issue of numerical artifacts through interpolation and refined triangulation. 
    To implement these extensive additions, the ToPoVis code is rewritten.
    The accuracy of the interpolation is measured for both linear and non-linear simulations.
    The new code is publicly avaliable as and can be used and extended in future research.
  ],
  acknowledgements: [
    I would like to express my gratitude to Dr. Florian Rath for his generous time and support.
    His infectious excitement and patience and openness for answering questions is what convinced me of writing this thesis in the first place and what helped me get through.

    I want to extend my thanks to Prof. Dr. Arthur for offering the opportunity of this thesis and taking his time for our weekly meetings.

    I would further like to thank everyone who has supported me during the time of this research:
    Leonard Schmid for enduring my infodumps and providing new valuable perspectives.
    Sara Kleyman for her encouragements and being my body double numerous time.
    Björn Luchterhandt for inspiring me to learn Typst and sharing his knowledge about it.
    Elli Quitter for always caring for and comforting me in times of need.

    Thanks to everyone that showed interest in this work and took the time to provide feedback.
  ]
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