# DiSE paper — ATVA draft

This directory contains a LaTeX draft of the DiSE paper, targeting
[ATVA](https://atva-conference.org/).

## Layout

```
paper/
  main.tex                   # Top-level document — \input's the sections
  references.bib             # Bibliography (splncs04 style)
  sections/
    00-abstract.tex
    01-introduction.tex
    02-problem-statement.tex
    03-related-work.tex
    04-algorithm.tex
    05-implementation.tex
    06-results.tex
    07-conclusion.tex
  figures/                   # convergence plots, frontier illustrations
  results/                   # numeric results pulled in by the tables
  README.md                  # this file
```

## Build

```bash
cd paper
latexmk -pdf main           # full build (BibTeX + multiple passes)
# Or manually:
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Requires the [`llncs`](https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines)
class file. Most TeX distributions ship it; if not, download
`llncs.cls` and `splncs04.bst` from the link above and place them
next to `main.tex`.

## Status

This is a working draft. Sections that need filling in before
submission:

- [ ] Empirical numbers in `06-results.tex` (Table 2): currently
      shows three benchmarks (coin_machine, gcd, assertion_overflow);
      run `scripts/reproduce.sh` to populate the rest.
- [ ] Convergence plot in `06-results.tex` (Figure 1): generate via
      `dise plot --kind convergence`.
- [ ] Soundness regression on assertion_overflow (§6.5): fix or
      mark as known limitation depending on follow-up work.
- [ ] Anonymisation of the repository URL.
- [ ] Author affiliation / ORCID.
- [ ] ATVA page-limit pass (currently uncapped — typical limit is
      16 + bibliography for the LNCS proceedings).

## Style guide

- One sentence per line in the source. Diffs are easier to read.
- Wrap lines at ~76 chars only where it doesn't hurt readability.
- Use `\Cref{}` for cross-references; `cleveref` is loaded.
- New macros go in `main.tex`; keep the list short.
- Citations: prefer the canonical conference/journal entry over
  arXiv when both exist. Verify dates before submission.
- Do **not** put numerical results directly in the prose;
  use the tables.
