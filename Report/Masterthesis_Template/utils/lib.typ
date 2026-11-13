#let report(
  title: "",
  subtitle: "",
  author: "",
  affiliation: "",
  date: datetime.today(),
  language: "en",
  abstract: none,
  keywords: none,
  bib-file: "refs.bib",
  logo: "fhtw_cover.png",
  body,
) = {
  set page(
    paper: "a4",
    margin: (top: 3cm, bottom: 3cm, inside: 3.5cm, outside: 2.5cm),
    header: none,
    footer: context {
      let page-num = counter(page).get().first()

      if page-num > 1 {
        let headings = query(heading.where(level: 1).before(here()))
        let left-text = if headings.len() > 0 { headings.last().body } else { title }
        text(9pt, fill: luma(100))[
          #left-text
          #h(1fr)
          #counter(page).display()
        ]
      }
    },
  )
  set text(font: "Liberation Sans", size: 11pt, lang: language)
  set par(justify: true, leading: 0.7em)

  page(
    header: none,
    footer: none,
    background: if logo != none {
      image(logo, width: 100%, height: 100%, fit: "stretch")
    },
  )[
    #v(1.5cm)
    #align(left)[
      #v(4cm)
      #text(22pt, weight: "bold")[#title]
      #if subtitle != "" [#v(0.5cm) #text(18pt)[#subtitle]]

      #v(3cm)
      #align(left)[
        #grid(
          columns: (auto, 1fr),
          row-gutter: 1.2em,
          column-gutter: 3.5em,
          [*Author:*], [#author],
          [*Affiliation:*], [#affiliation],
          [*Date:*], [#date.display("[day].[month].[year]")],
        )
      ]
    ]
  ]
  pagebreak()

  if abstract != none [
    #heading(level: 1, numbering: none)[Abstract]
    #abstract
    #if keywords != none [
      #v(0.4em)
      *Keywords:* #keywords
    ]
    #v(0.6em)
  ]

  set heading(numbering: "1.1")
  body

  if bib-file != none {
    bibliography(bib-file, style: "ieee")
  }
}