#!/bin/bash

export PYTHONPATH=.:../gate-lf-python-data
rm -rf ./doc-sphinx
rm -rf ./doc-html
mkdir doc-sphinx
mkdir doc-html
# cp sphinx-config/index.rst doc-sphinx
sphinx-apidoc -e -f -H "GATE LF Pytorch Wrapper (gatelfpytorch)" -A "Johann Petrak" -V "0.1" -o doc-sphinx gatelfpytorch
mv doc-sphinx/modules.rst doc-sphinx/index.rst
sphinx-build -b html -c sphinx-config doc-sphinx doc-html 
