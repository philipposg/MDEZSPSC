#!/bin/bash
# Download the knowledge graph, the class list and the test sets into data/.
# Zenodo record 23187515 (DOI 10.5281/zenodo.23187515); the files download once the record is published.
set -e
DATA_URL=${DATA_URL:-"https://zenodo.org/records/23187515/files"}
mkdir -p data/test
for f in graph_ilsvrc.pt ilsvrc_concepts.txt; do
    [ -f data/$f ] || wget -O data/$f "$DATA_URL/$f"
done
# test sets as image folders data/test/<set>/<state>/*.jpg (evaluate.py --test data/test/<set>)
for s in osdd vaw_states cgqa mit; do
    if [ ! -d data/test/$s ]; then
        wget -O data/test/$s.zip "$DATA_URL/$s.zip"
        unzip -q data/test/$s.zip -d data/test && rm data/test/$s.zip
    fi
done
