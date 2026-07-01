import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
import re
import numpy as np
import pysbd
import torch
from torch import nn
from TTS.config import load_config
from TTS.tts.models import setup_model as setup_tts_model
from TTS.utils.audio.numpy_transforms import save_wav
from TTS.utils.generic_utils import optional_to_str
if TYPE_CHECKING:
    from TTS.tts.models.base_tts import BaseTTS

logger = logging.getLogger(__name__)


PAD_SILENCE_SAMPLES = 10000

class Synthesizer(nn.Module):
    def __init__(
        self,
        *,
        tts_checkpoint: str | os.PathLike[Any] | None = None,
        tts_config_path: str | os.PathLike[Any] | None = None,
        encoder_checkpoint: str | os.PathLike[Any] | None = None,
        encoder_config: str | os.PathLike[Any] | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
        use_cuda: bool = False,
    ) -> None:
        super().__init__()
        self.tts_checkpoint = Path(optional_to_str(tts_checkpoint))
        self.tts_config_path = Path(tts_config_path) if tts_config_path is not None else None
        self.encoder_checkpoint = optional_to_str(encoder_checkpoint)
        self.encoder_config = optional_to_str(encoder_config)
        self.tts_model: BaseTTS | None = None
        self.seg = self._get_segmenter("en")
        self.default_language_name = "en"
        self.use_cuda = use_cuda

        if self.use_cuda:
            assert torch.cuda.is_available(), "CUDA is not availabe on this machine."

        checkpoint_dir = None
        if tts_checkpoint:
            self._load_tts(self.tts_checkpoint, self.tts_config_path, use_cuda=use_cuda)
            self.default_language_name = self._normalize_language_tag(
                getattr(self.tts_config, "phoneme_language", "en")
            )
            checkpoint_dir = self.tts_checkpoint if self.tts_checkpoint.is_dir() else self.tts_checkpoint.parent

        if checkpoint_dir is None:
            raise RuntimeError("Need to initialize a TTS model via tts_checkpoint")

        self.voice_dir = Path(voice_dir) if voice_dir is not None else checkpoint_dir / "voices"

    @staticmethod
    def _get_segmenter(lang: str) -> pysbd.Segmenter:

        return pysbd.Segmenter(language=lang, clean=True)

    def _load_tts(self, tts_checkpoint: Path, tts_config_path: Path | None = None, *, use_cuda: bool) -> None:

        checkpoint_dir = tts_checkpoint if tts_checkpoint.is_dir() else tts_checkpoint.parent
        if tts_config_path is None:
            tts_config_path = checkpoint_dir / "config.json"
        self.tts_config = load_config(tts_config_path)
        self.output_sample_rate = self.tts_config.audio.get("output_sample_rate", self.tts_config.audio["sample_rate"])
        if self.tts_config["use_phonemes"] and self.tts_config["phonemizer"] is None:
            msg = "Phonemizer is not defined in the TTS config."
            raise ValueError(msg)

        self.tts_model = setup_tts_model(config=self.tts_config)

        if not self.encoder_checkpoint and self.tts_config.model_args.get("speaker_encoder_config_path"):
            self.encoder_checkpoint = self.tts_config.model_args.speaker_encoder_model_path
            self.encoder_config = self.tts_config.model_args.speaker_encoder_config_path

        if tts_checkpoint.is_dir():
            # We assume the model knows how to load itself from a directory
            self.tts_model.load_checkpoint(self.tts_config, checkpoint_dir=tts_checkpoint, eval=True)
        else:
            self.tts_model.load_checkpoint(self.tts_config, checkpoint_path=tts_checkpoint, eval=True)
        if use_cuda:
            self.tts_model.cuda()

        if self.encoder_checkpoint and hasattr(self.tts_model, "speaker_manager"):
            self.tts_model.speaker_manager.init_encoder(self.encoder_checkpoint, self.encoder_config, use_cuda)


    @staticmethod
    def _normalize_language_tag(lang: str | None) -> str:
        if not lang:
            return "en"
        lang = str(lang).lower().strip()
        if lang.startswith("en"):
            return "en"
        if lang.startswith("vi"):
            return "vi"
        return lang

    @staticmethod
    def _split_vi_sentences(text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        parts = re.split(r"(?<=[.!?…;:])\s+", text)
        return [part.strip() for part in parts if part.strip()]

    def split_into_sentences(self, text: str, language_name: str | None = None) -> list[str]:
        lang = self._normalize_language_tag(language_name or self.default_language_name)
        if lang == "vi":
            return self._split_vi_sentences(text)
        return self.seg.segment(text)

    def save_wav(self, wav: list[int] | torch.Tensor | np.ndarray, path: str, pipe_out=None) -> None:

        # if tensor convert to numpy
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        if isinstance(wav, list):
            wav = np.array(wav)
        save_wav(wav=wav, path=path, sample_rate=self.output_sample_rate, pipe_out=pipe_out)


    def tts(
        self,
        text: str = "",
        speaker_name: str | None = "",
        language_name: str = "",
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
        *,
        split_sentences: bool = True,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> list[int] | dict[str, Any]:
        if self.tts_model is None:
            raise RuntimeError("Text-to-speech model not loaded")

        if not text:
            raise ValueError("You need to define `text` for synthesis.")

        start_time = time.time()
        segments = []
        current_time = 0.0
        wavs = []

        sens = [text]
        if split_sentences:
            sens = self.split_into_sentences(text, language_name=language_name)
            logger.info("Text split into sentences.")
        logger.info("Input: %s", sens)

        voice_dir = Path(d) if (d := kwargs.pop("voice_dir", None)) is not None else self.voice_dir

        for sen in sens:
            outputs = self.tts_model.synthesize(
                text=sen,
                speaker=speaker_name,
                voice_dir=voice_dir,
                speaker_wav=speaker_wav,
                language=language_name,
                use_griffin_lim=True,
                **kwargs,
            )
            waveform = outputs["wav"]
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            waveform = waveform.squeeze()

            if self.tts_config.audio.get("do_trim_silence"):
                waveform = waveform[: self.tts_model.ap.find_endpoint(waveform)]

            wavs += list(waveform)
            wavs += [0] * PAD_SILENCE_SAMPLES

            if return_dict:
                wav_duration_sec = len(waveform) / self.tts_config.audio["sample_rate"]
                segment = {
                    "id": len(segments),
                    "start": current_time,
                    "end": current_time + wav_duration_sec,
                    "text": sen,
                }
                segments.append(segment)
                current_time += wav_duration_sec
                current_time += PAD_SILENCE_SAMPLES / self.tts_config.audio["sample_rate"]

        process_time = time.time() - start_time
        audio_time = len(wavs) / self.tts_config.audio["sample_rate"]
        logger.info("Processing time: %.3f", process_time)
        logger.info("Real-time factor: %.3f", process_time / audio_time)

        if return_dict:
            return {"wav": wavs, "text": text, "segments": segments}
        return wavs