#!/usr/bin/env bash
# Fill in the GitHub user and the two Zenodo DOIs everywhere, then rebuild both PDFs.
#   tools/set_ids.sh <github_user> <software_doi_number> <preprint_doi_number>
#   example: tools/set_ids.sh jcdelrio 17000001 17000002
#   (the numbers are the part after "10.5281/zenodo.")
set -euo pipefail
[ $# -eq 3 ] || { echo "usage: $0 <github_user> <software_doi_number> <preprint_doi_number>"; exit 1; }
USER_GH=$1; SW=$2; PP=$3
[[ $SW =~ ^[0-9]+$ && $PP =~ ^[0-9]+$ ]] || { echo "DOI numbers must be digits only (e.g. 17000001)"; exit 1; }
cd "$(dirname "$0")/.."
FILES="README.md CITATION.cff paper/wosr_zenodo.tex paper/wosr_technical.tex"
sed -i.bak -e "s/GITHUB_USER/${USER_GH}/g" -e "s/SWDOI/${SW}/g" -e "s/PPDOI/${PP}/g" $FILES
rm -f README.md.bak CITATION.cff.bak paper/*.tex.bak
if command -v latexmk >/dev/null; then
  (cd paper && latexmk -pdf -interaction=nonstopmode -quiet wosr_zenodo.tex wosr_technical.tex && latexmk -c)
else
  echo "latexmk not found: compile paper/wosr_zenodo.tex and paper/wosr_technical.tex (pdflatex + bibtex) yourself"
fi
grep -rn "GITHUB_USER\|SWDOI\|PPDOI" $FILES && { echo "placeholders left!"; exit 1; } || echo "all identifiers set"
git status --short
