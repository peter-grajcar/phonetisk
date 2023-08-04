#!/bin/bash
set -e
mkdir -p models
cd models
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3278{/slovak-morfflex-pdt-170914.zip}
unzip slovak-morfflex-pdt-170914.zip
rm slovak-morfflex-pdt-170914.zip
