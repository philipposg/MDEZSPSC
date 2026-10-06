#!/bin/bash
# Upload the data files to the Zenodo draft (does NOT publish; publish from the web page after checking).
#   bash scripts/zenodo_upload.sh [file ...]      (default: data/graph_ilsvrc.pt data/ilsvrc_concepts.txt)
# Token: ~/.config/zenodo/token (chmod 600; never commit it or upload it to the record).
set -e
DEPOSITION=${DEPOSITION:-23187515}
TOKEN=$(cat ~/.config/zenodo/token)
API=https://zenodo.org/api/deposit/depositions/$DEPOSITION
FILES=("$@")
[ ${#FILES[@]} -eq 0 ] && FILES=(data/graph_ilsvrc.pt data/ilsvrc_concepts.txt)

BUCKET=$(curl -sf -H "Authorization: Bearer $TOKEN" "$API" | python3 -c 'import json,sys; print(json.load(sys.stdin)["links"]["bucket"])')
for f in "${FILES[@]}"; do
    echo "uploading $f ($(du -h "$f" | cut -f1)), md5 $(md5sum "$f" | cut -d' ' -f1)"
    # large files can fail transiently: retry, and show the server's answer on failure
    for attempt in 1 2 3; do
        out=$(curl -sS --upload-file "$f" -H "Authorization: Bearer $TOKEN" -w '\n%{http_code}' "$BUCKET/$(basename "$f")") || true
        code=${out##*$'\n'}
        [ "$code" = 201 ] || [ "$code" = 200 ] && break
        echo "  attempt $attempt failed (HTTP $code): ${out%$'\n'*}" | head -c 500; echo
    done
    [ "$code" = 201 ] || [ "$code" = 200 ] || exit 1
    echo "${out%$'\n'*}" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("  ->", d["key"], d["size"], d["checksum"])'
done
echo "files in the draft:"
curl -sf -H "Authorization: Bearer $TOKEN" "$API/files" \
    | python3 -c 'import json,sys; [print("  ", f["filename"], f["filesize"], f["checksum"]) for f in json.load(sys.stdin)]'
