import os
from typing import Any, Optional
import numpy as np
import torch
from coqpit import Coqpit

from TTS.tts.utils.managers import BaseIDManager


class LanguageManager(BaseIDManager):
    def __init__(
        self,
        language_ids_file_path: str | os.PathLike[Any] = "",
        config: Coqpit | None = None,
    ):
        super().__init__(id_file_path=language_ids_file_path)

        if config:
            self.set_language_ids_from_config(config)

    @property
    def num_languages(self) -> int:
        return len(list(self.name_to_id.keys()))

    @property
    def language_names(self) -> list[str]:
        return list(self.name_to_id.keys())

    @staticmethod
    def parse_language_ids_from_config(c: Coqpit) -> dict[str, int]:
        languages = set({})
        for dataset in c.datasets:
            if "language" in dataset:
                languages.add(dataset["language"])
            else:
                raise ValueError(f"Dataset {dataset['name']} has no language specified.")
        return {name: i for i, name in enumerate(sorted(languages))}

    def set_language_ids_from_config(self, c: Coqpit) -> None:
        self.name_to_id = self.parse_language_ids_from_config(c)

    @staticmethod
    def parse_ids_from_data(items: list[dict[str, Any]], parse_key: str) -> Any:
        raise NotImplementedError

    def set_ids_from_data(self, items: list[dict[str, Any]], parse_key: str) -> Any:
        raise NotImplementedError

    @staticmethod
    def init_from_config(config: Coqpit) -> Optional["LanguageManager"]:
        if config.model_args.get("use_language_embedding"):
            if config.model_args.get("language_ids_file"):
                return LanguageManager(language_ids_file_path=config.model_args.language_ids_file)
            return LanguageManager(config=config)
        return None


def get_language_balancer_weights(items: list[dict[str, Any]]) -> torch.Tensor:
    language_names = np.array([item["language"] for item in items])
    unique_language_names = np.unique(language_names).tolist()
    language_ids = [unique_language_names.index(l) for l in language_names]
    language_count = np.array([len(np.where(language_names == l)[0]) for l in unique_language_names])
    weight_language = 1.0 / language_count
    dataset_samples_weight = np.array([weight_language[l] for l in language_ids])
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    return torch.from_numpy(dataset_samples_weight).float()
