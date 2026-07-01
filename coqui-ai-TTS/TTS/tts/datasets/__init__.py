import logging
import os
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol
import numpy as np
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets.dataset import *
logger = logging.getLogger(__name__)

class Formatter(Protocol):
    def __call__(
        self,
        root_path: str | os.PathLike[Any],
        meta_file: str | os.PathLike[Any],
        ignored_speakers: list[str] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]: ...

_FORMATTER_REGISTRY: dict[str, Formatter] = {}

def register_formatter(name: str, formatter: Formatter) -> None:
    formatter_name = name.lower()
    if formatter_name in _FORMATTER_REGISTRY:
        msg = f"Formatter {name} already exists."
        raise ValueError(msg)
    _FORMATTER_REGISTRY[formatter_name] = formatter


def ljspeech(
    root_path: str | os.PathLike[Any],
    meta_file: str | os.PathLike[Any],
    ignored_speakers: list[str] | None = None,
    **kwargs,
) -> list[dict[str, Any]]:
    del ignored_speakers, kwargs

    root_path = Path(root_path)
    metadata_path = root_path / meta_file
    items: list[dict[str, Any]] = []
    speaker_name = "ljspeech"

    with open(metadata_path, encoding="utf-8") as metadata_file:
        for line_number, line in enumerate(metadata_file, start=1):
            line = line.strip()
            if not line:
                continue

            cols = line.split("|")
            if len(cols) < 3:
                msg = (
                    "LJSpeech format expects 3 pipe-delimited columns "
                    f"in {metadata_path}, line {line_number}."
                )
                raise IndexError(msg)

            wav_id = cols[0].strip()
            text = cols[2].strip()
            wav_file = root_path / "wavs" / f"{wav_id}.wav"

            items.append(
                {
                    "text": text,
                    "audio_file": str(wav_file),
                    "speaker_name": speaker_name,
                    "root_path": str(root_path),
                }
            )

    return items


register_formatter("ljspeech", ljspeech)


def split_dataset(
    items: list[dict[str, Any]], eval_split_max_size: int | None = None, eval_split_size: float = 0.01
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    speakers = [item["speaker_name"] for item in items]
    is_multi_speaker = len(set(speakers)) > 1
    if eval_split_size > 1:
        eval_split_size = int(eval_split_size)
    else:
        if eval_split_max_size:
            eval_split_size = min(eval_split_max_size, int(len(items) * eval_split_size))
        else:
            eval_split_size = int(len(items) * eval_split_size)

    assert eval_split_size > 0, (
        f"You do not have enough samples for the evaluation set. You can work around this setting the 'eval_split_size' parameter to a minimum of {1 / len(items)}"
    )
    np.random.seed(0)
    np.random.shuffle(items)
    if is_multi_speaker:
        items_eval = []
        speakers = [item["speaker_name"] for item in items]
        speaker_counter = Counter(speakers)
        while len(items_eval) < eval_split_size:
            item_idx = np.random.randint(0, len(items))
            speaker_to_be_removed = items[item_idx]["speaker_name"]
            if speaker_counter[speaker_to_be_removed] > 1:
                items_eval.append(items[item_idx])
                speaker_counter[speaker_to_be_removed] -= 1
                del items[item_idx]
        return items_eval, items
    return items[:eval_split_size], items[eval_split_size:]


def add_extra_keys(metadata: list[dict[str, Any]], language: str, dataset_name: str):
    for item in metadata:
        item["language"] = language
        relfilepath = Path(item["audio_file"]).relative_to(item["root_path"]).with_suffix("")
        item["audio_unique_name"] = f"{dataset_name}#{relfilepath}"
    return metadata


def load_tts_samples(
    datasets: list[BaseDatasetConfig] | BaseDatasetConfig,
    eval_split: bool = True,
    formatter: Formatter | None = None,
    eval_split_max_size: int | None = None,
    eval_split_size: float = 0.01,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    
    meta_data_train_all = []
    meta_data_eval_all = []
    if not isinstance(datasets, list):
        datasets = [datasets]
    for dataset in datasets:
        formatter_name = dataset["formatter"]
        dataset_name = dataset["dataset_name"]
        root_path = dataset["path"]
        meta_file_train = dataset["meta_file_train"]
        meta_file_val = dataset["meta_file_val"]
        ignored_speakers = dataset["ignored_speakers"]
        language = dataset["language"]
        if formatter is None:
            formatter = _get_formatter_by_name(formatter_name)
        meta_data_train = formatter(root_path, meta_file_train, ignored_speakers=ignored_speakers)
        assert len(meta_data_train) > 0, f" [!] No training samples found in {root_path}/{meta_file_train}"

        meta_data_train = add_extra_keys(meta_data_train, language, dataset_name)

        logger.info("Found %d files in %s", len(meta_data_train), Path(root_path).resolve())
        if eval_split:
            if meta_file_val:
                meta_data_eval = formatter(root_path, meta_file_val, ignored_speakers=ignored_speakers)
                meta_data_eval = add_extra_keys(meta_data_eval, language, dataset_name)
            else:
                eval_size_per_dataset = eval_split_max_size // len(datasets) if eval_split_max_size else None
                meta_data_eval, meta_data_train = split_dataset(meta_data_train, eval_size_per_dataset, eval_split_size)
            meta_data_eval_all += meta_data_eval
        meta_data_train_all += meta_data_train
        if dataset.meta_file_attn_mask:
            meta_data = dict(load_attention_mask_meta_data(dataset["meta_file_attn_mask"]))
            for meta_data_all in (meta_data_train_all, meta_data_eval_all):
                for idx, ins in enumerate(meta_data_all):
                    attn_file = meta_data[ins["audio_file"]].strip()
                    meta_data_all[idx].update({"alignment_file": attn_file})
        formatter = None
    return meta_data_train_all, meta_data_eval_all


def load_attention_mask_meta_data(metafile_path: str | os.PathLike[Any]):
    with open(metafile_path, encoding="utf-8") as f:
        lines = f.readlines()

    meta_data = []
    for line in lines:
        wav_file, attn_file = line.split("|")
        meta_data.append([wav_file, attn_file])
    return meta_data


def _get_formatter_by_name(name: str) -> Formatter:
    if name.lower() not in _FORMATTER_REGISTRY:
        msg = f"{name} formatter not found. If it is a custom formatter, make sure to call register_formatter() first."
        raise ValueError(msg)
    return _FORMATTER_REGISTRY[name.lower()]


def find_unique_chars(data_samples: list[dict[str, Any]]) -> set[str]:
    texts = "".join(item["text"] for item in data_samples)
    chars = set(texts)
    lower_chars = filter(lambda c: c.islower(), chars)
    chars_force_lower = [c.lower() for c in chars]
    chars_force_lower = set(chars_force_lower)
    logger.info("Number of unique characters: %d", len(chars))
    logger.info("Unique characters: %s", "".join(sorted(chars)))
    logger.info("Unique lower characters: %s", "".join(sorted(lower_chars)))
    logger.info("Unique all forced to lower characters: %s", "".join(sorted(chars_force_lower)))
    return chars_force_lower
