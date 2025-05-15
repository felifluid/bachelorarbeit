#import "template.typ": apply-template
#import "@preview/numbly:0.1.0": numbly

#show: apply-template(
  title: "Improved Visualization of Turbulences in a Tokamak using refined Triangulation and Interpolation",
  author: "Feli Nara Celeste",
  city: "Bayreuth",
  submissionDate: "TBD" // TODO: add this
)[

#include "chapters/abstract.typ"

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