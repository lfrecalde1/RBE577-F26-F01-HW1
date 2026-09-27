# Rebuild the manuscript without re-running experiments.
.PHONY: all diagram clean

all: diagram
	latexmk -pdf -interaction=nonstopmode -halt-on-error ship_allocation_report.tex

diagram: figures/network_diagram.pdf

figures/network_diagram.pdf: figures/network_diagram.tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=figures figures/network_diagram.tex

clean:
	latexmk -c ship_allocation_report.tex
	latexmk -c -outdir=figures figures/network_diagram.tex
