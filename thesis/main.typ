#import "template.typ": apply-template

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

#set heading(numbering: "A.", supplement: [Appendix])
#counter(heading).update(0)

#include "chapters/usage.typ"

]