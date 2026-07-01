from dataclasses import asdict, dataclass, field
from pathlib import Path
from coqpit import Coqpit, check_argument
from TTS.config.shared_configs import (
    BaseAudioConfig,
    BaseDatasetConfig,
    BaseTrainingConfig,
)
from TTS.config.shared_configs import ModelArgs


@dataclass
class GSTConfig(Coqpit):
    gst_style_input_wav: str = None
    gst_style_input_weights: dict = None
    gst_embedding_dim: int = 256
    gst_use_speaker_embedding: bool = False
    gst_num_heads: int = 4
    gst_num_style_tokens: int = 10
    def check_values(self) -> None:
        c = asdict(self)
        super().check_values()
        check_argument("gst_style_input_weights", c, restricted=False)
        check_argument("gst_style_input_wav", c, restricted=False)
        check_argument("gst_embedding_dim", c, restricted=True, min_val=0, max_val=1000)
        check_argument("gst_use_speaker_embedding", c, restricted=False)
        check_argument("gst_num_heads", c, restricted=True, min_val=2, max_val=10)
        check_argument("gst_num_style_tokens", c, restricted=True, min_val=1, max_val=1000)


@dataclass
class CapacitronVAEConfig(Coqpit):
    capacitron_loss_alpha: int = 1
    capacitron_capacity: int = 150
    capacitron_VAE_embedding_dim: int = 128
    capacitron_use_text_summary_embeddings: bool = True
    capacitron_text_summary_embedding_dim: int = 128
    capacitron_use_speaker_embedding: bool = False
    capacitron_VAE_loss_alpha: float = 0.25
    capacitron_grad_clip: float = 5.0
    def check_values(self) -> None:
        c = asdict(self)
        super().check_values()
        check_argument("capacitron_capacity", c, restricted=True, min_val=10, max_val=500)
        check_argument("capacitron_VAE_embedding_dim", c, restricted=True, min_val=16, max_val=1024)
        check_argument("capacitron_use_speaker_embedding", c, restricted=False)
        check_argument("capacitron_text_summary_embedding_dim", c, restricted=False, min_val=16, max_val=512)
        check_argument("capacitron_VAE_loss_alpha", c, restricted=False)
        check_argument("capacitron_grad_clip", c, restricted=False)


@dataclass
class CharactersConfig(Coqpit):
    characters_class: str = None
    vocab_dict: list[str] | None = None
    pad: str = "<PAD>"
    eos: str = None
    bos: str = None
    blank: str = None
    characters: str = None
    punctuations: str = None
    phonemes: str = None
    is_unique: bool = True  
    is_sorted: bool = True


@dataclass
class BaseTTSConfig(BaseTrainingConfig):
    audio: BaseAudioConfig = field(default_factory=BaseAudioConfig)
    model_args: ModelArgs = field(default_factory=ModelArgs)
    _supports_cloning: bool = False
    use_phonemes: bool = False
    phonemizer: str = None
    phoneme_language: str = None
    compute_input_seq_cache: bool = False
    text_cleaner: str = None
    enable_eos_bos_chars: bool = False
    test_sentences_file: str = ""
    phoneme_cache_path: str = None
    characters: CharactersConfig = None
    add_blank: bool = False
    batch_group_size: int = 0
    loss_masking: bool = None
    min_audio_len: int = 1
    max_audio_len: int = float("inf")
    min_text_len: int = 1
    max_text_len: int = float("inf")
    compute_f0: bool = False
    compute_energy: bool = False
    compute_linear_spec: bool = False
    precompute_num_workers: int = 0
    use_noise_augment: bool = False
    start_by_longest: bool = False
    shuffle: bool = False
    drop_last: bool = False
    datasets: list[BaseDatasetConfig] = field(default_factory=list)
    optimizer: str = "radam"
    optimizer_params: dict = None
    lr_scheduler: str = None
    lr_scheduler_params: dict = field(default_factory=dict)
    test_sentences: list[str] | list[list[str]] = field(default_factory=list)
    eval_split_max_size: int = None
    eval_split_size: float = 0.01
    use_speaker_weighted_sampler: bool = False
    speaker_weighted_sampler_alpha: float = 1.0
    use_language_weighted_sampler: bool = False
    language_weighted_sampler_alpha: float = 1.0
    use_length_weighted_sampler: bool = False
    length_weighted_sampler_alpha: float = 1.0

    @property
    def supports_cloning(self) -> bool:
        return self._supports_cloning or (
            Path(self.model_args.get("speaker_encoder_model_path", "")).is_file()
            and Path(self.model_args.get("speaker_encoder_config_path", "")).is_file()
        )
