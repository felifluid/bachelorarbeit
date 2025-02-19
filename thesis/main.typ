#import "functions.typ": load-bib

#set heading()

// TODO: Deckblatt

#outline(
    indent: 1em,
)

#pagebreak()

#include "chapters/abstract.typ"

#include "chapters/introduction.typ"

#include "chapters/background.typ"

#include "chapters/topovis-refinement.typ"

#include "chapters/code.typ"

#include "chapters/conclusion.typ"

#include "chapters/outlook.typ"

#load-bib(main: true)