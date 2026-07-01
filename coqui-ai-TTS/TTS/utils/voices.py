import datetime
import importlib.metadata
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from TTS.utils.generic_utils import is_pytorch_at_least_2_4
except ImportError:
    def is_pytorch_at_least_2_4():
        return False

try:
    from TTS.utils.generic_utils import slugify
except ImportError:
    def slugify(text):
        return str(text).strip().replace(" ", "_").replace("/", "_")

logger = logging.getLogger(__name__)


@dataclass
class VoiceMetadata:
    model: dict[str, str | float | bool]
    speaker_id: str
    source_files: list[str] | None = None
    created_at: str | None = None
    coqui_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceMetadata":
        return cls(**data)


class CloningMixin:
    def _create_voice_metadata(
        self, model: dict[str, str | float | bool], speaker_id: str, source_files: list[str]
    ) -> VoiceMetadata:
        return VoiceMetadata(
            model=model,
            speaker_id=speaker_id,
            source_files=source_files,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes"),
            coqui_version=importlib.metadata.version("coqui-tts"),
        )

    def clone_voice(
        self,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None,
        speaker_id: str | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
        **generate_kwargs: Any,
    ) -> dict[str, Any]:
        if speaker_wav is None or (isinstance(speaker_wav, list) and len(speaker_wav) == 0):
            if speaker_id is None:
                msg = "Neither `speaker_wav` nor `speaker_id` was specified"
                raise RuntimeError(msg)
            if voice_dir is None:
                msg = "Specified only `speaker_id`, but no `voice_dir` to load the voice from"
                raise RuntimeError(msg)
            return self.load_voice_file(speaker_id, voice_dir)
        voice, model_metadata = self._clone_voice(speaker_wav, **generate_kwargs)
        logger.info("Generated voice from reference audio")
        if speaker_id is not None and voice_dir is not None:
            speaker_id = slugify(speaker_id)
            voice_fn = Path(voice_dir) / f"{speaker_id}.pth"
            voice_fn.parent.mkdir(exist_ok=True, parents=True)
            speaker_wav = speaker_wav if isinstance(speaker_wav, list) else [speaker_wav]
            metadata = self._create_voice_metadata(model_metadata, speaker_id, [str(p) for p in speaker_wav])
            voices = self.get_voices(voice_dir)
            if speaker_id in voices:
                logger.info("Voice `%s` already exists in `%s`, overwriting it", speaker_id, voice_fn)
            voice_dict = {**voice, "metadata": metadata.to_dict()}
            torch.save(voice_dict, voice_fn)
            logger.info("Voice `%s` saved to: %s", speaker_id, voice_fn)
        return voice

    def _clone_voice(
        self,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]],
        **generate_kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raise NotImplementedError

    def load_voice_file(
        self,
        speaker_id: str,
        voice_dir: str | os.PathLike[Any],
    ) -> dict[str, Any]:
        voices = self.get_voices(voice_dir)
        if speaker_id not in voices:
            msg = f"Voice file `{slugify(speaker_id)}.pth` for speaker `{speaker_id}` not found in: {voice_dir}"
            raise FileNotFoundError(msg)
        voice = torch.load(voices[speaker_id], map_location="cpu", weights_only=is_pytorch_at_least_2_4())
        logger.info("Loaded voice `%s` from: %s", speaker_id, voices[speaker_id])
        return voice

    def get_voices(self, voice_dir: str | os.PathLike[Any]) -> dict[str, Path]:
        return {path.stem: path for path in Path(voice_dir).glob("*.pth")}
