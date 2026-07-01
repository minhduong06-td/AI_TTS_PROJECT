import logging
import os
import random
from typing import Any, Literal, cast
import torch
import torch.distributed as dist
from coqpit import Coqpit
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from trainer.logging.base_dash_logger import BaseDashboardLogger
from trainer.torch import DistributedSampler, DistributedSamplerWrapper
from TTS.config import get_from_config_or_model_args
from TTS.config.shared_configs import ModelArgs
from TTS.model import BaseTrainerModel
from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.datasets.dataset import TTSDataset
from TTS.tts.utils.data import get_length_balancer_weights
from TTS.tts.utils.languages import LanguageManager, get_language_balancer_weights
from TTS.tts.utils.speakers import SpeakerManager, get_speaker_balancer_weights
from TTS.tts.utils.synthesis import inv_spectrogram
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
try:
    from TTS.utils.generic_utils import warn_synthesize_config_deprecated, warn_synthesize_speaker_id_deprecated
except ImportError:
    def warn_synthesize_config_deprecated(*args, **kwargs):
        return None

    def warn_synthesize_speaker_id_deprecated(*args, **kwargs):
        return None
from TTS.utils.voices import CloningMixin
logger = logging.getLogger(__name__)


class BaseTTS(CloningMixin, BaseTrainerModel):
    MODEL_TYPE = "tts"
    config: BaseTTSConfig

    def __init__(
        self,
        config: Coqpit,
        ap: "AudioProcessor",
        tokenizer: "TTSTokenizer",
        speaker_manager: SpeakerManager | None = None,
        language_manager: LanguageManager | None = None,
    ):
        super().__init__()
        self.config = cast(BaseTTSConfig, config)
        self.ap = ap
        self.tokenizer = tokenizer
        self.speaker_manager = speaker_manager
        self.language_manager = language_manager
        self._set_model_args()

    def _set_model_args(self) -> None:
        if isinstance(self.config, BaseTTSConfig):
            config_num_chars = get_from_config_or_model_args(self.config, "num_chars")
            num_chars = config_num_chars if self.tokenizer is None else self.tokenizer.characters.num_chars
            if "characters" in self.config:
                self.config.num_chars = num_chars
                self.config.model_args.num_chars = num_chars
            self.args = self.config.model_args
        elif isinstance(self.config, ModelArgs):
            self.args = self.config
        else:
            raise ValueError("config must be either a *Config or *Args")

    def init_multispeaker(self, config: Coqpit):
        if self.speaker_manager is not None:
            self.num_speakers = self.speaker_manager.num_speakers
        elif hasattr(config, "num_speakers"):
            self.num_speakers = config.num_speakers
        if config.use_speaker_embedding or config.use_d_vector_file:
            self.embedded_speaker_dim = (
                config.d_vector_dim if "d_vector_dim" in config and config.d_vector_dim is not None else 512
            )
        if config.use_speaker_embedding and not config.use_d_vector_file:
            logger.info("Init speaker_embedding layer.")
            self.speaker_embedding = nn.Embedding(self.num_speakers, self.embedded_speaker_dim)
            self.speaker_embedding.weight.data.normal_(0, 0.3)

    def get_aux_input_from_test_sentences(self, sentence_info: str | list[str]) -> dict[str, Any]:
        text, speaker, style_wav, language = None, None, None, None

        if isinstance(sentence_info, list):
            if len(sentence_info) == 1:
                text = sentence_info[0]
            elif len(sentence_info) == 2:
                text, speaker = sentence_info
            elif len(sentence_info) == 3:
                text, speaker, style_wav = sentence_info
            elif len(sentence_info) == 4:
                text, speaker, style_wav, language = sentence_info
        else:
            text = sentence_info

        if speaker is None and self.speaker_manager is not None:
            speaker = random.sample(self.speaker_manager.speaker_names, 1)[0]

        return {
            "text": text,
            "speaker": speaker,
            "style_wav": style_wav,
            "language": language,
        }

    def format_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        text_input = batch["token_id"]
        text_lengths = batch["token_id_lengths"]
        speaker_names = batch["speaker_names"]
        linear_input = batch["linear"]
        mel_input = batch["mel"]
        mel_lengths = batch["mel_lengths"]
        stop_targets = batch["stop_targets"]
        item_idx = batch["item_idxs"]
        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]
        attn_mask = batch["attns"]
        waveform = batch["waveform"]
        pitch = batch["pitch"]
        energy = batch["energy"]
        language_ids = batch["language_ids"]
        max_text_length = torch.max(text_lengths.float())
        max_spec_length = torch.max(mel_lengths.float())
        durations = None
        if attn_mask is not None:
            durations = torch.zeros(attn_mask.shape[0], attn_mask.shape[2])
            for idx, am in enumerate(attn_mask):
                c_idxs = am[:, : text_lengths[idx], : mel_lengths[idx]].max(1)[1]
                c_idxs, counts = torch.unique(c_idxs, return_counts=True)
                dur = torch.ones([text_lengths[idx]]).to(counts.dtype)
                dur[c_idxs] = counts
                extra_frames = dur.sum() - mel_lengths[idx]
                largest_idxs = torch.argsort(-dur)[:extra_frames]
                dur[largest_idxs] -= 1
                assert dur.sum() == mel_lengths[idx], (
                    f" [!] total duration {dur.sum()} vs spectrogram length {mel_lengths[idx]}"
                )
                durations[idx, : text_lengths[idx]] = dur
        stop_targets = stop_targets.view(text_input.shape[0], stop_targets.size(1) // self.config.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze(2)
        stop_target_lengths = torch.divide(mel_lengths, self.config.r).ceil_()

        return {
            "text_input": text_input,
            "text_lengths": text_lengths,
            "speaker_names": speaker_names,
            "mel_input": mel_input,
            "mel_lengths": mel_lengths,
            "linear_input": linear_input,
            "stop_targets": stop_targets,
            "stop_target_lengths": stop_target_lengths,
            "attn_mask": attn_mask,
            "durations": durations,
            "speaker_ids": speaker_ids,
            "d_vectors": d_vectors,
            "max_text_length": float(max_text_length),
            "max_spec_length": float(max_spec_length),
            "item_idx": item_idx,
            "waveform": waveform,
            "pitch": pitch,
            "energy": energy,
            "language_ids": language_ids,
            "audio_unique_names": batch["audio_unique_names"],
        }

    def get_sampler(self, config: Coqpit, dataset: TTSDataset, num_gpus=1):
        weights = None
        data_items = dataset.samples

        if getattr(config, "use_language_weighted_sampler", False):
            alpha = getattr(config, "language_weighted_sampler_alpha", 1.0)
            logger.info("Using Language weighted sampler with alpha: %.2f", alpha)
            weights = get_language_balancer_weights(data_items) * alpha

        if getattr(config, "use_speaker_weighted_sampler", False):
            alpha = getattr(config, "speaker_weighted_sampler_alpha", 1.0)
            logger.info("Using Speaker weighted sampler with alpha: %.2f", alpha)
            if weights is not None:
                weights += get_speaker_balancer_weights(data_items) * alpha
            else:
                weights = get_speaker_balancer_weights(data_items) * alpha

        if getattr(config, "use_length_weighted_sampler", False):
            alpha = getattr(config, "length_weighted_sampler_alpha", 1.0)
            logger.info("Using Length weighted sampler with alpha: %.2f", alpha)
            if weights is not None:
                weights += get_length_balancer_weights(data_items) * alpha
            else:
                weights = get_length_balancer_weights(data_items) * alpha

        if weights is not None:
            sampler = WeightedRandomSampler(weights, len(weights))
        else:
            sampler = None
        if sampler is None:
            sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        else:  
            sampler = DistributedSamplerWrapper(sampler) if num_gpus > 1 else sampler

        return sampler

    def get_data_loader(
        self,
        config: Coqpit,
        assets: dict,
        is_eval: bool,
        samples: list[dict] | list[list],
        verbose: bool,
        num_gpus: int,
        rank: int | None = None,
    ) -> "DataLoader":
        if self.speaker_manager is not None:
            speaker_id_mapping = (
                self.speaker_manager.name_to_id
                if get_from_config_or_model_args(config, "use_speaker_embedding")
                else None
            )
            d_vector_mapping = (
                self.speaker_manager.embeddings if get_from_config_or_model_args(config, "use_d_vector_file") else None
            )
            config.use_d_vector_file = get_from_config_or_model_args(config, "use_d_vector_file", False)
        else:
            speaker_id_mapping = None
            d_vector_mapping = None
        if self.language_manager is not None:
            language_id_mapping = self.language_manager.name_to_id if self.args.use_language_embedding else None
        else:
            language_id_mapping = None
        dataset = TTSDataset(
            outputs_per_step=config.r if "r" in config else 1,
            compute_linear_spec=config.model.lower() == "tacotron" or config.compute_linear_spec,
            compute_f0=config.get("compute_f0", False),
            f0_cache_path=config.get("f0_cache_path", None),
            compute_energy=config.get("compute_energy", False),
            energy_cache_path=config.get("energy_cache_path", None),
            samples=samples,
            ap=self.ap,
            return_wav=config.return_wav if "return_wav" in config else False,
            batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
            min_text_len=config.min_text_len,
            max_text_len=config.max_text_len,
            min_audio_len=config.min_audio_len,
            max_audio_len=config.max_audio_len,
            phoneme_cache_path=config.phoneme_cache_path,
            precompute_num_workers=config.precompute_num_workers,
            use_noise_augment=False if is_eval else config.use_noise_augment,
            speaker_id_mapping=speaker_id_mapping,
            d_vector_mapping=d_vector_mapping if config.use_d_vector_file else None,
            tokenizer=self.tokenizer,
            start_by_longest=config.start_by_longest,
            language_id_mapping=language_id_mapping,
        )

        if num_gpus > 1:
            dist.barrier()
        dataset.preprocess_samples()
        sampler = self.get_sampler(config, dataset, num_gpus)

        return DataLoader(
            dataset,
            batch_size=config.eval_batch_size if is_eval else config.batch_size,
            shuffle=config.shuffle if sampler is None else False,  # if there is no other sampler
            collate_fn=dataset.collate_fn,
            drop_last=config.drop_last,  
            sampler=sampler,
            num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
            pin_memory=False,
        )

    def _create_logs(
        self, batch: dict[str, Any], outputs: dict[str, Any] | list[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raise NotImplementedError

    @torch.inference_mode()
    def train_log(
        self,
        batch: dict[str, Any],
        outputs: dict[str, Any] | list[dict[str, Any]],
        logger: BaseDashboardLogger,
        assets: dict[str, Any],
        steps: int,
    ) -> None:
        figures, audios = self._create_logs(batch, outputs)
        logger.train_figures(steps, figures)
        logger.train_audios(steps, audios, self.ap.sample_rate)

    @torch.inference_mode()
    def eval_step(
        self, batch: dict[str, Any], criterion: nn.Module, optimizer_idx: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if optimizer_idx is not None:
            return self.train_step(batch, criterion, optimizer_idx)
        return self.train_step(batch, criterion)

    @torch.inference_mode()
    def eval_log(
        self,
        batch: dict[str, Any],
        outputs: dict[str, Any] | list[dict[str, Any]],
        logger: BaseDashboardLogger,
        assets: dict[str, Any],
        steps: int,
    ) -> None:
        figures, audios = self._create_logs(batch, outputs)
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.ap.sample_rate)

    @torch.inference_mode()
    def test_run(self, assets: dict) -> dict[str, Any]:
        logger.info("Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        if len(test_sentences) == 0:
            logger.warning("No test sentences provided.")
        for idx, sen in enumerate(test_sentences):
            aux_inputs = self.get_aux_input_from_test_sentences(sen)
            outputs = self.synthesize(
                aux_inputs["text"],
                speaker=aux_inputs.get("speaker", None),
                language=aux_inputs.get("language", None),
                use_griffin_lim=True,
            )
            test_audios[f"{idx}-audio"] = outputs["wav"]
            test_figures[f"{idx}-prediction"] = plot_spectrogram(
                outputs["outputs"]["model_outputs"], self.ap, output_fig=False
            )
            test_figures[f"{idx}-alignment"] = plot_alignment(outputs["alignments"], output_fig=False)
        return {"figures": test_figures, "audios": test_audios}

    def test_log(
        self,
        outputs: dict[str, Any],
        logger: "Logger",
        assets: dict,
        steps: int,  
    ) -> None:
        logger.test_audios(steps, outputs["audios"], self.ap.sample_rate)
        if "figures" in outputs:
            logger.test_figures(steps, outputs["figures"])

    def on_init_start(self, trainer):
        if self.speaker_manager is not None:
            output_path = os.path.join(trainer.output_path, "speakers.pth")
            self.speaker_manager.save_ids_to_file(output_path)
            trainer.config.speakers_file = output_path
            trainer.config.model_args.speakers_file = output_path
            trainer.config.save_json(os.path.join(trainer.output_path, "config.json"))
            logger.info("`speakers.pth` is saved to: %s", output_path)
            logger.info("`speakers_file` is updated in the config.json.")

        if self.language_manager is not None:
            output_path = os.path.join(trainer.output_path, "language_ids.json")
            self.language_manager.save_ids_to_file(output_path)
            trainer.config.model_args.language_ids_file = output_path
            trainer.config.save_json(os.path.join(trainer.output_path, "config.json"))
            logger.info("`language_ids.json` is saved to: %s", output_path)
            logger.info("`language_ids_file` is updated in the config.json.")

    def _get_language_id(self, language: str | None) -> int | None:
        if self.language_manager is not None:
            if len(self.language_manager.name_to_id) == 1:
                return list(self.language_manager.name_to_id.values())[0]
            if language is not None:
                try:
                    return self.language_manager.name_to_id[language]
                except KeyError as e:
                    msg = (
                        f"Looks like you use a multi-lingual model. "
                        f"Language {language} is not among the available languages: "
                        f"{self.language_manager.name_to_id.keys()}."
                    )
                    raise ValueError(msg) from e
            msg = "Looks like you use a multi-lingual model, but did not specify a language. "
            raise ValueError(msg)
        return None

    def _get_speaker_id_or_dvector(
        self,
        speaker: str | None,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.speaker_manager is None:
            return None, None

        if len(self.speaker_manager.name_to_id) == 1:
            speaker_id = list(self.speaker_manager.name_to_id.values())[0]
            return torch.tensor([speaker_id], device=self.device), None

        speaker_exists = True
        if get_from_config_or_model_args(self.config, "use_d_vector_file") and speaker is not None:
            if speaker in self.speaker_manager.embedding_names:
                d_vector = self.speaker_manager.get_mean_embedding(speaker, num_samples=None, randomize=False)
                d_vector = torch.tensor(d_vector, dtype=torch.float, device=self.device).unsqueeze(0)
                return None, d_vector  # [1 x embedding_dim]
            speaker_exists = False

        if get_from_config_or_model_args(self.config, "use_speaker_embedding") and speaker is not None:
            if speaker in self.speaker_manager.name_to_id:
                speaker_id = self.speaker_manager.name_to_id[speaker]
                return torch.tensor([speaker_id], device=self.device), None
            speaker_exists = False

        if self.speaker_manager.encoder is not None and (speaker is not None or speaker_wav is not None):
            d_vector = self.clone_voice(speaker_wav, speaker, voice_dir)["d_vector"]
            return None, torch.tensor(d_vector, dtype=torch.float, device=self.device).unsqueeze(0)

        if not speaker_exists:
            msg = f"{speaker} is not a valid speaker of the model."
            raise KeyError(msg)

        msg = (
            "Looks like you are using a multi-speaker model. "
            "You need to pass either a speaker name or a reference audio file."
        )
        raise ValueError(msg)

    def _get_speaker_conditioning(
        self,
        aux_input: dict[str, Any],
        spk_emb_module_name: str = "speaker_embedding",
        *,
        normalize_d_vector: bool = True,
        normalize_embedding: bool = True,
        output_shape: Literal["BCT", "BTC"] = "BCT",
    ) -> torch.Tensor | None:
        sid = aux_input.get("speaker_ids")
        g = aux_input.get("d_vectors")

        if sid is not None and g is not None:
            msg = "Cannot use both speaker_ids and d_vectors simultaneously. "
            raise ValueError(msg)

        if g is not None and normalize_d_vector:
            g = F.normalize(g)
        if get_from_config_or_model_args(self.config, "use_speaker_embedding") and sid is not None:
            spk_emb_module = getattr(self, spk_emb_module_name, None)
            if spk_emb_module is None:
                msg = "Speaker embedding requested but no embedding module provided"
                raise RuntimeError(msg)
            g = spk_emb_module(sid)
            assert g.ndim == 2
            if normalize_embedding:
                g = F.normalize(g)
        if g is not None:
            if output_shape == "BCT":
                g = g.unsqueeze(-1)
            elif output_shape == "BTC":
                g = g.unsqueeze(1)
            else:
                msg = f"Invalid output shape `{output_shape}`. Use `BCT` or `BTC`."
                raise ValueError(msg)
        return g

    def _clone_voice(
        self, speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]], **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        d_vector = self.speaker_manager.compute_embedding_from_clip(speaker_wav)
        voice = {"d_vector": d_vector}
        metadata = {"name": self.speaker_manager.encoder.__class__.__name__}
        return voice, metadata

    def synthesize(
        self,
        text: str,
        config: BaseTTSConfig | None = None,
        *,
        speaker: str | None = None,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
        language: str | None = None,
        use_griffin_lim: bool = False,
        do_trim_silence: bool = False,
        extra_aux_input: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if config is not None:
            warn_synthesize_config_deprecated()
        if (speaker_id := kwargs.pop("speaker_id", None)) is not None:
            speaker = speaker_id
            warn_synthesize_speaker_id_deprecated()
        text_inputs = self.tokenizer.text_to_ids(text, language=language)
        language_id = self._get_language_id(language)
        _speaker_id, d_vector = self._get_speaker_id_or_dvector(speaker, speaker_wav, voice_dir)
        text_inputs = torch.as_tensor(text_inputs, dtype=torch.long, device=self.device).unsqueeze(0)
        if language_id is not None:
            language_id = torch.tensor([language_id], device=self.device)

        if extra_aux_input is None:
            extra_aux_input = {}
        outputs = self.inference(
            text_inputs,
            aux_input={
                "x_lengths": torch.tensor(text_inputs.shape[1:2], device=self.device),
                "speaker_ids": _speaker_id,
                "d_vectors": d_vector,
                "language_ids": language_id,
                **extra_aux_input,
            },
        )
        model_outputs = outputs["model_outputs"]
        model_outputs = model_outputs[0].detach().to(torch.float32).cpu().numpy().squeeze()
        alignments = outputs["alignments"]

        wav = None
        if model_outputs.ndim == 2:  
            if use_griffin_lim:
                wav = inv_spectrogram(model_outputs, self.ap, self.config)
                if do_trim_silence:
                    wav = wav[: self.ap.find_endpoint(wav)]
        else: 
            wav = model_outputs
        return {
            "wav": wav,
            "alignments": alignments,
            "text_inputs": text_inputs,
            "outputs": outputs,
        }


class BaseTTSE2E(BaseTTS):
    def _set_model_args(self) -> None:
        if isinstance(self.config, BaseTTSConfig):
            num_chars = (
                self.config.model_args.num_chars if self.tokenizer is None else self.tokenizer.characters.num_chars
            )
            self.config.model_args.num_chars = num_chars
            self.config.num_chars = num_chars
            self.args = self.config.model_args
            self.args.num_chars = num_chars
        elif isinstance(self.config, ModelArgs):
            self.args = self.config
        else:
            raise ValueError("config must be either a *Config or *Args")
