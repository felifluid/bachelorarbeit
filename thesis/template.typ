#import "@preview/hydra:0.6.1": hydra
#import "@preview/equate:0.3.1": equate
#import "@preview/numbly:0.1.0": numbly

#let convert_arr(arr) = {
    str(arr)
}

#let apply-template(
    documentName: [Thesis],
    title: [Title],
    author: "Author",
    faculty: "Faculty / Group",
    degree: "Degree",
    supervisors: ("Name1", "Name 2"),
    city: "City",
    submissionDate: datetime(day:1, month: 1, year: 1970),
    disclaimerTitle: "Disclaimer",
    disclaimerText: [
        I hereby declare that I prepared this thesis entirely on my own and have not used outside sources without declaration in the text.
        Any concepts or quotations applicable to these sources are clearly attributed to them.
        This thesis has not been submitted in the same or substantially similar version, not even in part, to any other authority for grading and has not been published elsewhere.
    ],
    disclaimerSignature: none,
    body
    ) = context {
        // DOCUMENT
        set document(author: author, title: title)

        // TEXT
        set text(lang: "en", size: 12pt, ligatures: false)

        // TITLE PAGE
        set page(numbering: none)

        align(center)[
            // TODO: Logo
            #faculty

            //title
            #text(2em, weight: 700, title)

            #v(1fr)

            in Partial Fulfillment of the Requirements for the \
            Degree of \
            #text(1.5em, degree)

            #v(1fr)
            
            // author
            by \
            #smallcaps(text(1.25em, author))

            #v(1fr)

            supervized by \
            #text(1.25em, supervisors.at(0)) \
            and \
            #text(1.25em, supervisors.at(1))

            #v(1fr)

            #city, #submissionDate.display("[day].[month].[year]")
        ]

        // FONTS
        set text(font: "New Computer Modern")

        // PAGE
        // set header
        set page(
            // display current chapter as page header
            header: context {
                align(center, smallcaps(hydra(1)))
            },
        )
        
        // PAR
        set par(justify: true)

        // HEADINGS
        // clean numbering (https://sitandr.github.io/typst-examples-book/book/snippets/numbering.html?highlight=numbering#clean-numbering)
        
        set heading(
            numbering: numbly(
                "{1:I}",
                "{2:1}",
                "{2:1}.{3:1}",
                "{2:1}.{3:1}.{4:1}",
        ))

        show heading: it => {
            let lvl = it.level
            if lvl == 1 {
                block(
                    width: 100%, 
                    above: 1.95em,
                    below: 1em
                )[
                    #set align(center)
                    #set text(size: 1.25em)
                    #smallcaps(it)
                ]
            } else if lvl == 2 {
                block(
                    above: 1.85em,
                    below: 1em,
                )[
                    #set text(size: 1.15em)
                    #it
                ]
            } else if lvl == 3 {
                block(
                    above: 1.75em,
                    below: 1em
                )[
                    #set text(size: 1.10em)
                    #it
                ]
            } else if lvl == 4 {
                block(
                    above: 1.55em,
                    below: 1em
                )[
                    #set text(size: 1.05em)
                    #it
                ]
            } else if lvl >= 5 {
                block(
                    above: 1.55em,
                    below: 1em
                )[
                    #set text(size: 1.0em)
                    // omit numbering
                    #it.body
                ]
            } 
        }

        // page break before new chapter
        show heading.where(level: 1): it => [
            #pagebreak(weak: true)
            #it
        ]

        // MATH EQUATIONS
        // per chapter numbering
        show heading.where(level:1): it => {
            counter(math.equation).update(0)
            it
        }

        let math_numbering(..n) = {
                numbering("(I.1)", counter(heading).get().first(), ..n)
        }

        set math.equation(numbering: math_numbering)

        // only number labeled equations
        show: equate.with(breakable: true, sub-numbering: true, number-mode: "label")

        // don't linebreak inline equations
        show math.equation.where(block: false): box


        // adjust page numbering
        set page(numbering: "i")
        counter(page).update(1)

        // TABLE OF CONTENTS
        let custom_indent(n) = {
            if n > 1 {
                n*1em
            } else {
                0em
            }
        }

        set outline(indent: custom_indent)
        set outline.entry(fill: repeat(".", gap: 0.5em))

        show outline.entry.where(level: 1): set block(above: 1em)
        show outline.entry.where(level: 1): it => {
            link(it.element.location(), strong(it.indented(it.prefix(), it.body())))
        }

        show outline.entry.where(level: 1): set text(size:1.15em)
        show outline.entry.where(level: 2): set text(size:1em)
        show outline.entry.where(level: 3): set text(size:0.95em)
        show outline.entry.where(level: 4): set text(size:0.9em)

        outline(
            title: "Contents",
            depth: 4,
        )

        //TODO: add preface

        // PAGE NUMBERING
        set page(numbering: "1")
        counter(page).update(1)

        set cite(
            style: "institute-of-electrical-and-electronics-engineers",
        )

        body
        
        // BIBLIOGRAPHY
        set page(numbering: none)
        bibliography("bibliography.yml")
        
        // DISCLAIMER
        set heading(numbering: none)

        heading(disclaimerTitle, outlined: false)

        disclaimerText

        v(3em)
        grid(
          columns: 2,
          column-gutter: 1fr,
          row-gutter: 0.5em,
          align: center,
          grid.cell(
            rowspan: 3,
            [#city, #submissionDate.display("[day].[month].[year]")],
          ),
          disclaimerSignature,
          line(length: 6cm, stroke: 0.5pt),
          author,
        )

}