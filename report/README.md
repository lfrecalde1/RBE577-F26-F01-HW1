# Ship allocation report

The report consists of `ship_allocation_report.tex`, `references.bib`, the
PDF figures in `figures/`, and the numerical values and tables in `generated/`.
The compiled document is `ship_allocation_report.pdf`.

## Build and edit

From the package root:

```bash
make -C report
```

Or, from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error ship_allocation_report.tex
```

Building uses the saved figures and tables. It does not require Python,
training, or access to the datasets. A TeX installation with IEEEtran,
amsmath, amssymb, bm, graphicx, booktabs, array, xcolor, balance, and hyperref
is required. TikZ and standalone are needed to rebuild the network diagram.
Use `make -C report clean` from the package root to remove intermediate files.

Edit text and equations in the main `.tex` file, references in `references.bib`,
and table rows in `generated/`. The network diagram has an editable TikZ source
at `figures/network_diagram.tex`. Other figures are supplied as vector PDFs.

## Numerical content

The figures and tables retain the reported million-sample results. Its split
was 800,000 training, 100,000 validation, and 100,000 evaluation samples.
The raw datasets, checkpoints, experiment drivers, and validation scripts are
not part of this report directory. Its LaTeX build uses the saved assets directly.

The main package programs create new datasets, train a network, report test-set
metrics, and compare NLP, neural allocation, and direct forces. See the
[package README](../README.md) for that workflow.

`ship_allocation_report_source.zip` contains only the current manuscript and
the assets needed to build it. The package README link above applies to the
workspace copy.
