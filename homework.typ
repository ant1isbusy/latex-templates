
#set heading(numbering: "1.")
#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2cm),
  numbering: "1",
)

#set par(justify: true)

#set text(
  size: 12pt,
)

#show title: set text(size: 18pt)
#show title: set align(center)

#show math.equation.where(block: true): eq => {
  block(width: 100%, inset: 0pt, align(center, eq))
}

#title[
  Course Assignment X
]
#align(center)[
  Firstname Lastname \
  Mat. Nr.: ....]


#show heading: set text(
  size: 14pt,
  weight: "semibold",
)
#show heading: smallcaps

= Heading

