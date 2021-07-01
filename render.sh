#!/bin/bash

mkdir -p "rendered-pdf"
mkdir -p "rendered-html"

CSS_PATH="styles/vue.css"

for f in lecture-*.md; do
  html_filename="./rendered-html/${f%.*}.html"

  echo "Converting $f to $html_filename"
  pandoc --standalone --mathjax --css "$CSS_PATH" -o "$html_filename" -- "$f"

  pdf_filename="./rendered-pdf/${f%.*}.pdf"
  echo "Converting $f to $pdf_filename.pdf"
  pandoc --katex --css "$CSS_PATH" -f markdown -t html5 -o "$pdf_filename" -- "$f"

  sed -i "" "s|images/|../images/|g" "$html_filename"
done
