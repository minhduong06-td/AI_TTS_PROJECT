from __future__ import annotations
import os
from typing import Iterable

def _resolve_audio_path(root_path: str, wav_id: str) -> str:
    wav_id = wav_id.strip()
    candidates: list[str] = []
    if os.path.isabs(wav_id):
        candidates.append(wav_id)
    else:
        candidates.append(os.path.join(root_path, wav_id))

    if not wav_id.lower().endswith(".wav"):
        candidates.append(os.path.join(root_path, "wavs", f"{wav_id}.wav"))
        candidates.append(os.path.join(root_path, f"{wav_id}.wav"))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    return candidates[1] if len(candidates) > 1 else candidates[0]


def ljspeech(root_path: str, meta_file: str, ignored_speakers: Iterable[str] | None = None):
    ignored_speakers = set(ignored_speakers or [])
    speaker_name = "ljspeech"
    if speaker_name in ignored_speakers:
        return []

    metadata_path = meta_file if os.path.isabs(meta_file) else os.path.join(root_path, meta_file)
    items = []

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            cols = line.split("|")
            if len(cols) < 2:
                raise ValueError(f"Invalid LJSpeech metadata line {line_no}: {line!r}")

            wav_id = cols[0].strip()
            text = (cols[2] if len(cols) > 2 and cols[2].strip() else cols[1]).strip()
            audio_file = _resolve_audio_path(root_path, wav_id)

            items.append(
                {
                    "text": text,
                    "audio_file": audio_file,
                    "speaker_name": speaker_name,
                    "root_path": root_path,
                }
            )

    return items
