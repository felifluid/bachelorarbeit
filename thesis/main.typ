#import "template.typ": apply-template

#show: apply-template(
  author: "Feli Nara Celeste",
  city: "Bayreuth",
  submissionDate: "" // TODO: add this
)[

#include "chapters/abstract.typ"

#include "chapters/introduction.typ"

#include "chapters/background.typ"

#include "chapters/topovis-refinement.typ"

#include "chapters/conclusion.typ"

#include "chapters/outlook.typ"

// Addendum

#set heading(numbering: "A)", supplement: [Appendix])
#counter(heading).update(0)

#include "chapters/usage.typ"

]