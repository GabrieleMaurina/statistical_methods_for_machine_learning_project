#!/usr/bin/env bash

export TEXINPUTS=..//:
pdflatex --shell-escape report &&
bibtex report.aux &&
pdflatex --shell-escape report &&
pdflatex --shell-escape report
rm -rf report.aux report.log _minted-report report.toc report.bbl report.blg report.out
xdg-open report.pdf
