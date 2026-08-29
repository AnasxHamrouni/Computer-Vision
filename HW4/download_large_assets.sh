#!/usr/bin/env bash
# Downloads the HW4 Drive folder (zip + demo videos).
# Requires: pip install gdown
set -euo pipefail
FOLDER_ID="1u__uYojc0tdCILr5VsDkYwwH3azczJ1z"
DEST="${1:-./_drive_assets}"

mkdir -p "$DEST"
echo "Downloading Google Drive folder into $DEST ..."
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "$DEST"

echo
echo "Next:"
echo "  1. Unzip 3DGSexploration_01.zip"
echo "  2. Copy ConferenceHall_unpacked.ply → scenes/conference-hall/"
echo "  3. Optionally copy outdoor-standard.ply → this directory"
