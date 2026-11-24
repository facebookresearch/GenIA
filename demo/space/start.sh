#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# The Space's entry point: fetch the model weights, then serve the demo app.
#
# SAM 3D Objects and SAM3 are gated on Hugging Face: the Space needs the secret HF_TOKEN, a
# token of an account whose access to both was granted. With persistent storage (/data)
# the weights, the uploads and the runs outlive a restart; without it they are fetched
# again each time the Space starts.
set -e
#: Where the recorded examples are published (demo/space/replays_gcs.py push).
REPLAYS_URL=https://storage.googleapis.com/geniadata/demo/replays
eval "$("$HOME/conda/bin/conda" shell.bash hook)"
conda activate genia
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -d /data ] && [ -w /data ]; then
    export HF_HOME=/data/huggingface TORCH_HOME=/data/torch
    WORK=/data/demo_app
else
    WORK=$HOME/demo_app
fi
mkdir -p "$WORK"
# The recorded examples, from Google Cloud Storage into the app's work dir (only what
# changed since the last start); without them the examples simply run live.
python demo/space/replays_gcs.py pull "${GENIA_REPLAYS_URL:-$REPLAYS_URL}" "$WORK/replays" \
    || echo "No recorded examples fetched: the examples run live."

python -m genia.core.download_weights --no-actionmesh
exec python demo/app.py --work-dir "$WORK"
