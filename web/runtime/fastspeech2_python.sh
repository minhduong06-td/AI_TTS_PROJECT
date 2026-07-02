#!/usr/bin/env bash
set -e

ARGS="$*"

# ============================================================
# FastSpeech2 checkpoints của bạn dùng vocab khác nhau:
# EN checkpoint embedding = 361  => cần symbols = 360
# VI checkpoint embedding = 803  => cần symbols = 802
# Wrapper này tự set vocab theo dataset trong args.
# ============================================================

if [[ "$ARGS" == *"en_ljspeech"* ]]; then
  export FASTSPEECH2_SYMBOL_COUNT=360
elif [[ "$ARGS" == *"vi_ljspeech"* ]]; then
  export FASTSPEECH2_SYMBOL_COUNT=802
else
  echo "ERROR: Không xác định được ngôn ngữ FastSpeech2 từ args:" >&2
  echo "$ARGS" >&2
  exit 1
fi

export PYTHONPATH="/home/md_dz6/AI/FastSpeech2/src:${PYTHONPATH}"

exec python3 "$@"
