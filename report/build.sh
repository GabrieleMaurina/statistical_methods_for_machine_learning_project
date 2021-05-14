#!/usr/bin/env bash

export TEXINPUTS=..//:
pdflatex --shell-escape report &&
pdflatex --shell-escape report
rm -rf report.aux report.log _minted-report report.toc
xdg-open report.pdf
