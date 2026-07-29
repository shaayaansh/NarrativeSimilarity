# ACL LaTeX Draft

This folder contains an ACL-style LaTeX scaffold for the NarrativeSimilarity write-up.

## Structure

- `main.tex`: main ACL-style paper file.
- `acl.sty`: official ACL style file from `acl-org/acl-style-files`.
- `acl_natbib.bst`: official ACL bibliography style file from `acl-org/acl-style-files`.
- `references.bib`: bibliography stub.
- `sections/`: one file per requested paper section.
- `figures/`, `tables/`: placeholders for paper assets.

## Compile

From this directory:

```bash
latexmk -pdf main.tex
```

For camera-ready mode, remove `[review]` from `\usepackage[review]{acl}` in `main.tex`.
