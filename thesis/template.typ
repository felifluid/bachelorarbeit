#import "@preview/hydra:0.6.1": hydra
#import "@preview/equate:0.3.1": equate

#let convert_arr(arr) = {
    str(arr)
}

#let apply-template(
    documentName: [Thesis],
    title: [Title],
    author: "Author",
    city: "City",
    submissionDate: "Submission Date",
    disclaimerTitle: "Disclaimer",
    disclaimerText: [
        I hereby declare that I prepared this thesis entirely on my own and have not used outside sources without declaration in the text.
        Any concepts or quotations applicable to these sources are clearly attributed to them.
        This thesis has not been submitted in the same or substantially similar version, not even in part, to any other authority for grading and has not been published elsewhere.
    ],
    body
    ) = context {
        // DOCUMENT
        set document(author: author, title: title)

        // TEXT
        set text(lang: "en", size: 12pt, ligatures: false)

        // TITLE PAGE
        set page(numbering: none)

        align(center)[
            //title
            #text(2em, weight: 700, title)

            #v(1fr)
            
            // author
            by \
            #smallcaps(text(1.25em, author))

            #v(1fr)

            #city, #submissionDate
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
        // let heading_numbering(..schemes) = {
        //     (..nums) => {
        //         let (section, ..subsections) = nums.pos()
        //         let (section_scheme, ..subschemes) = schemes.pos()
        // 
        //         if subsections.len() == 0 {
        //             numbering(section_scheme, section)
        //         } else if subschemes.len() == 0 {
        //             numbering(section_scheme, ..nums.pos())
        //         } else {
        //             heading_numbering(..subschemes)(..subsections)
        //         }
        //     }
        // }
        // 
        // set heading(numbering: heading_numbering("I.", "1."))
        
        set heading(numbering: "I.1.1")
        
        show heading: it => {
            if it.level > 1 {
                let counter = counter(heading).get().map(str)
                let nums = counter.slice(1)
        
                let num = "0"
        
                if nums.len() > 1{
                    num = nums.join(".")
                } else {
                    num = nums.at(0)
                }
                num + " " + it.body
            } else {
                it
            }
        }

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
        outline(
            title: "Contents",
            indent: 2em,
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
}