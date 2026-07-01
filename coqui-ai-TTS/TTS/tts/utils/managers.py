import json
import os
import random
from typing import Any
import fsspec
import numpy as np
import torch

from TTS.config import load_config
try:
    from TTS.encoder.models.base_encoder import BaseEncoder
    from TTS.encoder.utils.generic_utils import setup_encoder_model
except ModuleNotFoundError:
    BaseEncoder = Any
    setup_encoder_model = None
from TTS.utils.audio import AudioProcessor
try:
    from TTS.utils.generic_utils import is_pytorch_at_least_2_4
except ImportError:
    def is_pytorch_at_least_2_4():
        return False


def load_file(path: str | os.PathLike[Any]) -> Any:
    path = str(path)
    if path.endswith(".json"):
        with fsspec.open(path, "r") as f:
            return json.load(f)
    elif path.endswith(".pth"):
        with fsspec.open(path, "rb") as f:
            return torch.load(f, map_location="cpu", weights_only=is_pytorch_at_least_2_4())
    else:
        raise ValueError("Unsupported file type")


def save_file(obj: Any, path: str | os.PathLike[Any]):
    path = str(path)
    if path.endswith(".json"):
        with fsspec.open(path, "w") as f:
            json.dump(obj, f, indent=4)
    elif path.endswith(".pth"):
        with fsspec.open(path, "wb") as f:
            torch.save(obj, f)
    else:
        raise ValueError("Unsupported file type")


class BaseIDManager:
    def __init__(self, id_file_path: str | os.PathLike[Any] = ""):
        self.name_to_id: dict[str, int] = {}

        if id_file_path:
            self.load_ids_from_file(id_file_path)

    def set_ids_from_data(self, items: list[dict[str, Any]], parse_key: str) -> None:
        self.name_to_id = self.parse_ids_from_data(items, parse_key=parse_key)

    def load_ids_from_file(self, file_path: str | os.PathLike[Any]) -> None:
        self.name_to_id = load_file(file_path)

    def save_ids_to_file(self, file_path: str | os.PathLike[Any]) -> None:
        save_file(self.name_to_id, file_path)

    def get_random_id(self) -> int | None:
        if self.name_to_id:
            return self.name_to_id[random.choices(list(self.name_to_id.keys()))[0]]
        return None

    @staticmethod
    def parse_ids_from_data(items: list[dict[str, Any]], parse_key: str) -> dict[str, int]:
        classes = sorted({item[parse_key] for item in items})
        ids = {name: i for i, name in enumerate(classes)}
        return ids


class EmbeddingManager(BaseIDManager):
    def __init__(
        self,
        embedding_file_path: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
        id_file_path: str | os.PathLike[Any] = "",
        encoder_model_path: str | os.PathLike[Any] = "",
        encoder_config_path: str | os.PathLike[Any] = "",
        use_cuda: bool = False,
    ):
        super().__init__(id_file_path=id_file_path)

        self.embeddings = {}
        self.embeddings_by_names = {}
        self.clip_ids = []
        self.encoder: BaseEncoder | None = None
        self.encoder_ap: AudioProcessor | None = None
        self.use_cuda = use_cuda

        if embedding_file_path:
            if isinstance(embedding_file_path, list):
                self.load_embeddings_from_list_of_files(embedding_file_path)
            else:
                self.load_embeddings_from_file(embedding_file_path)

        if encoder_model_path and encoder_config_path:
            self.init_encoder(encoder_model_path, encoder_config_path, use_cuda)

    @property
    def num_embeddings(self) -> int:
        return len(self.embeddings)

    @property
    def num_names(self) -> int:
        return len(self.embeddings_by_names)

    @property
    def embedding_dim(self) -> int:
        if self.embeddings:
            return len(self.embeddings[list(self.embeddings.keys())[0]]["embedding"])
        return 0

    @property
    def embedding_names(self) -> list[str]:
        return list(self.embeddings_by_names.keys())

    def save_embeddings_to_file(self, file_path: str | os.PathLike[Any]) -> None:
        save_file(self.embeddings, file_path)

    @staticmethod
    def read_embeddings_from_file(
        file_path: str | os.PathLike[Any],
    ) -> tuple[dict[str, int], list[str], Any, dict[str, Any]]:
        embeddings = load_file(file_path)
        speakers = sorted({x["name"] for x in embeddings.values()})
        name_to_id = {name: i for i, name in enumerate(speakers)}
        clip_ids = list(set(clip_name for clip_name in embeddings.keys()))
        # cache embeddings_by_names for fast inference using a bigger speakers.json
        embeddings_by_names = {}
        for x in embeddings.values():
            if x["name"] not in embeddings_by_names.keys():
                embeddings_by_names[x["name"]] = [x["embedding"]]
            else:
                embeddings_by_names[x["name"]].append(x["embedding"])
        return name_to_id, clip_ids, embeddings, embeddings_by_names

    def load_embeddings_from_file(self, file_path: str | os.PathLike[Any]) -> None:
        self.name_to_id, self.clip_ids, self.embeddings, self.embeddings_by_names = self.read_embeddings_from_file(
            file_path
        )

    def load_embeddings_from_list_of_files(self, file_paths: list[str | os.PathLike[Any]]) -> None:
        self.name_to_id = {}
        self.clip_ids = []
        self.embeddings_by_names = {}
        self.embeddings = {}
        for file_path in file_paths:
            ids, clip_ids, embeddings, embeddings_by_names = self.read_embeddings_from_file(file_path)
            duplicates = set(self.embeddings.keys()) & set(embeddings.keys())
            if duplicates:
                raise ValueError(f"Duplicate embedding names <{duplicates}> in {file_path}")
            self.name_to_id.update(ids)
            self.clip_ids.extend(clip_ids)
            self.embeddings_by_names.update(embeddings_by_names)
            self.embeddings.update(embeddings)

        # reset name_to_id to get the right speaker ids
        self.name_to_id = {name: i for i, name in enumerate(self.name_to_id)}

    def get_embedding_by_clip(self, clip_idx: str) -> list:
        return self.embeddings[clip_idx]["embedding"]

    def get_embeddings_by_name(self, idx: str) -> list[list]:
        return self.embeddings_by_names[idx]

    def get_embeddings_by_names(self) -> dict[str, Any]:
        embeddings_by_names = {}
        for x in self.embeddings.values():
            if x["name"] not in embeddings_by_names.keys():
                embeddings_by_names[x["name"]] = [x["embedding"]]
            else:
                embeddings_by_names[x["name"]].append(x["embedding"])
        return embeddings_by_names

    def get_mean_embedding(self, idx: str, num_samples: int | None = None, randomize: bool = False) -> np.ndarray:
        embeddings = self.get_embeddings_by_name(idx)
        if num_samples is None:
            embeddings = np.stack(embeddings).mean(0)
        else:
            assert len(embeddings) >= num_samples, f" [!] {idx} has number of samples < {num_samples}"
            if randomize:
                embeddings = np.stack(random.choices(embeddings, k=num_samples)).mean(0)
            else:
                embeddings = np.stack(embeddings[:num_samples]).mean(0)
        return embeddings

    def get_random_embedding(self) -> Any:
        if self.embeddings:
            return self.embeddings[random.choices(list(self.embeddings.keys()))[0]]["embedding"]

        return None

    def get_clips(self) -> list[str]:
        return sorted(self.embeddings.keys())

    def init_encoder(
        self, model_path: str | os.PathLike[Any], config_path: str | os.PathLike[Any], use_cuda: bool = False
    ) -> None:
        if setup_encoder_model is None:
            raise RuntimeError("TTS.encoder is not available in this trimmed local-only build.")

        self.use_cuda = use_cuda
        self.encoder_config = load_config(config_path)
        self.encoder = setup_encoder_model(self.encoder_config)
        self.encoder_criterion = self.encoder.load_checkpoint(
            self.encoder_config, str(model_path), eval=True, use_cuda=use_cuda, cache=True
        )
        self.encoder_ap = AudioProcessor(**self.encoder_config.audio)

    @torch.inference_mode()
    def compute_embedding_from_clip(
        self, wav_file: str | os.PathLike[Any] | list[str | os.PathLike[Any]]
    ) -> list[float]:
        def _compute(wav_file: str | os.PathLike[Any]) -> torch.Tensor:
            if self.encoder_ap is None or self.encoder is None:
                msg = "You must first initialize the encoder with init_encoder()"
                raise RuntimeError(msg)
            waveform = self.encoder_ap.load_wav(wav_file, sr=self.encoder_ap.sample_rate)
            if not self.encoder_config.model_params.get("use_torch_spec", False):
                m_input = self.encoder_ap.melspectrogram(waveform)
                m_input = torch.from_numpy(m_input)
            else:
                m_input = torch.from_numpy(waveform)

            if self.use_cuda:
                m_input = m_input.cuda()
            m_input = m_input.unsqueeze(0)
            embedding = self.encoder.compute_embedding(m_input)
            return embedding

        if isinstance(wav_file, list):
            # compute the mean embedding
            embeddings = torch.stack([_compute(wf) for wf in wav_file])
            return embeddings.mean(dim=0)[0].tolist()
        embedding = _compute(wav_file)
        return embedding[0].tolist()

    def compute_embeddings(self, feats: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        if feats.ndim == 2:
            feats = feats.unsqueeze(0)
        if self.use_cuda:
            feats = feats.cuda()
        if self.encoder is None:
            msg = "You must first initialize the encoder with init_encoder()"
            raise RuntimeError(msg)
        return self.encoder.compute_embedding(feats)
