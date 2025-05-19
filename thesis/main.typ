#import "template.typ": apply-template
#import "@preview/numbly:0.1.0": numbly

#show: apply-template(
  title: "Improved Visualization of Turbulences in a Tokamak using refined Triangulation and Interpolation",
  faculty: "Faculty of Theoretical Plasma Physics \ TPV",
  author: "Feli Nara Celeste Berner",
  supervisors: ("Prof. Dr. Arthur Peeters", "Dr. Florian Rath"),
  degree: "Bachelor of Science (LA Gym)",
  city: "Bayreuth",
  submissionDate: datetime.today(),
  preface: [
    ToPoVis is a code that computes and visualizes plasma structures as poloidal cross sections in global tokamak geometry.
    This thesis tackles the issue of numerical artifacts through interpolation and refined triangulation. 
    To implement these extensive additions, the ToPoVis code is rewritten.
    The accuracy of the interpolation is measured for both linear and non-linear simulations.
    The new code is publicly avaliable as and can be used and extended in future research.
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