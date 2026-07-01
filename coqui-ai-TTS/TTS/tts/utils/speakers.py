import logging
import os
from typing import Any, Union
import numpy as np
import torch
from coqpit import Coqpit
from TTS.config import get_from_config_or_model_args
from TTS.tts.utils.managers import EmbeddingManager
logger = logging.getLogger(__name__)


class SpeakerManager(EmbeddingManager):
    def __init__(
        self,
        data_items: list[dict[str, Any]] | None = None,
        d_vectors_file_path: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
        speaker_id_file_path: str | os.PathLike[Any] = "",
        encoder_model_path: str | os.PathLike[Any] = "",
        encoder_config_path: str | os.PathLike[Any] = "",
        use_cuda: bool = False,
    ):
        super().__init__(
            embedding_file_path=d_vectors_file_path,
            id_file_path=speaker_id_file_path,
            encoder_model_path=encoder_model_path,
            encoder_config_path=encoder_config_path,
            use_cuda=use_cuda,
        )

        if data_items:
            self.set_ids_from_data(data_items, parse_key="speaker_name")

    @property
    def num_speakers(self) -> int:
        return len(self.name_to_id)

    @property
    def speaker_names(self) -> list[str]:
        return list(self.name_to_id.keys())

    @staticmethod
    def init_from_config(
        config: "Coqpit", samples: list[dict[str, Any]] | None = None
    ) -> Union["SpeakerManager", None]:
        speaker_manager = None
        if get_from_config_or_model_args(config, "use_speaker_embedding"):
            if samples:
                speaker_manager = SpeakerManager(data_items=samples)
            if speaker_file := get_from_config_or_model_args(config, "speaker_file"):
                speaker_manager = SpeakerManager(speaker_id_file_path=speaker_file)
            if speakers_file := get_from_config_or_model_args(config, "speakers_file"):
                speaker_manager = SpeakerManager(speaker_id_file_path=speakers_file)

        if get_from_config_or_model_args(config, "use_d_vector_file"):
            speaker_manager = SpeakerManager()
            if d_vector_file := get_from_config_or_model_args(config, "d_vector_file"):
                speaker_manager = SpeakerManager(d_vectors_file_path=d_vector_file)
        return speaker_manager


def get_speaker_balancer_weights(items: list):
    speaker_names = np.array([item["speaker_name"] for item in items])
    unique_speaker_names = np.unique(speaker_names).tolist()
    speaker_ids = [unique_speaker_names.index(l) for l in speaker_names]
    speaker_count = np.array([len(np.where(speaker_names == l)[0]) for l in unique_speaker_names])
    weight_speaker = 1.0 / speaker_count
    dataset_samples_weight = np.array([weight_speaker[l] for l in speaker_ids])
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    return torch.from_numpy(dataset_samples_weight).float()
