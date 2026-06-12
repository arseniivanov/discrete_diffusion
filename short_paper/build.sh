#!/usr/bin/env bash
# Compile the ACL paper. Runs the standard pdflatex -> bibtex -> pdflatex x2
# sequence (4 LaTeX passes total) so that citations and cross-references resolve.
#
# Usage:  ./build.sh            # build acl_latex.pdf
#         ./build.sh clean      # remove build artifacts
set -euo pipefail

JOB="acl_latex"
cd "$(dirname "$0")"

if [[ "${1:-}" == "clean" ]]; then
  rm -f "$JOB".{aux,bbl,blg,log,out,pdf,toc}
  echo "Cleaned build artifacts."
  exit 0
fi

if ! command -v pdflatex >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: pdflatex was not found on this machine.

The minimal texlive-base install here does not ship the LaTeX format or the
article class / fonts that acl.sty needs. Install the LaTeX packages with:

  sudo apt-get update
  sudo apt-get install -y texlive-latex-base texlive-latex-recommended \
       texlive-latex-extra texlive-fonts-recommended texlive-science

(You can run that from this Claude Code session with a leading "! ".)
Then re-run ./build.sh
EOF
  exit 1
fi

# 1st pass: write .aux with \citation and \bibcite keys
pdflatex -interaction=nonstopmode -halt-on-error "$JOB".tex
# resolve the bibliography (acl.sty sets \bibliographystyle{acl_natbib})
bibtex "$JOB"
# 2nd + 3rd passes: pull in .bbl and settle all cross-references
pdflatex -interaction=nonstopmode -halt-on-error "$JOB".tex
pdflatex -interaction=nonstopmode -halt-on-error "$JOB".tex

echo
echo "Built $JOB.pdf"
