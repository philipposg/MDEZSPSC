#!/bin/bash
# Download the knowledge graph and the class lists into data/.
# TODO before release: upload graph_ilsvrc.pt and set DATA_URL (e.g. a Zenodo record).
set -e
DATA_URL=${DATA_URL:-"https://zenodo.org/record/XXXXXXX/files"}
mkdir -p data
for f in graph_ilsvrc.pt ilsvrc_concepts.txt; do
    [ -f data/$f ] || wget -O data/$f "$DATA_URL/$f"
done
echo "Test sets: see README (data/test/<osdd|cgqa|mit|vaw_states>.h5)."
