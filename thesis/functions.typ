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