#!/bin/bash
set -e
mkdir -p models
cd models
curl -o "slovak-morfflex-pdt-170914.zip" "https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/handle/11234/1-3278/slovak-morfflex-pdt-170914.zip"
unzip slovak-morfflex-pdt-170914.zip
rm slovak-morfflex-pdt-170914.zip
