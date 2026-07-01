#!/usr/bin/env bash
set -e

cd /home/md_dz6/AI/web/be

export FASTSPEECH_PYTHON="${FASTSPEECH_PYTHON:-/home/md_dz6/AI/web/runtime/fastspeech2_python.sh}"
export COQUI_PYTHON="${COQUI_PYTHON:-/home/md_dz6/micromamba/envs/coqui/bin/python}"
export USE_CUDA="${USE_CUDA:-1}"
export NLTK_DATA="/home/md_dz6/nltk_data"
python3 app.py
