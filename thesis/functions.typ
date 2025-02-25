#let load-bib(main: false) = {
  let bib = "bibliography.yml"

  counter("bibs").step()

  context if main {
    [#bibliography(bib) <main-bib>]
  } else if query(<main-bib>) == () and counter("bibs").get().first() == 1 {
    // This is the first bibliography, and there is no main bibliography
    bibliography(bib)
  }
}

#let enum_numbering(..schemes) = {
  (..nums) => {
    let (enum, ..subenums) = nums.pos()
    let (enum_schema, ..subschemes) = schemes.pos()

    if subenums.len() == 0 {
      numbering(enum_schema, enum)
    } else if subenums.len() == 0 {
      numbering(enum_schema, ..nums.pos())
    }
    else {
      enum_numbering(..subschemes)(..subenums)
    }
  }
}