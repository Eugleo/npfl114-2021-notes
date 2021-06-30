#!/bin/bash

mkdir -p output
for f in *.md; do
  filename="./output/${f%.*}"
  echo "Converting $f to $filename.html"
  pandoc --css "/Users/eugen/Library/Application Support/abnerworks.Typora/themes/vue.css" -o "$filename.html" -- "$f"
done
