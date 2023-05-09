#let project(title: "", subtitle: "", authors: (), body) = {

  set document(author: authors.map(a => a.name), title: title)
  set page(numbering: "1", number-align: center)

  // set text(11pt, font: "Barlow", weight: 400)
  show raw: text.with(11pt, font: "Inconsolata")
  // show heading: text.with(14pt, font: "Inter", weight: 700)
  // show math.equation: set text(font: "Fira Math")

  set heading(numbering: "1.1)")
  show link: underline
  show ref: underline

  align(center)[
	// #block(text(font: "Inter", weight: 700, 1.75em, title))
	// #block(text(font: "Inter", weight: 400,  1.25em, subtitle))
  #block(text(weight: 700, 1.75em, title))
	#block(text(weight: 400,  1.25em, subtitle))
  ]

  pad(
	top: 0.5em,
	bottom: 0.5em,
	x: 2em,
	grid(
	  columns: (1fr,) * calc.min(3, authors.len()),
    gutter: 1em,
    ..authors.map(author => align(center)[
		*#author.name* \
		#author.email
    ]),
	),
  )

  // Main body.
  set par(justify: true)

  body
}

