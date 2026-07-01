from dataclasses import dataclass, field
from coqpit import Coqpit
from TTS.config.shared_configs import ModelArgs
from TTS.tts.configs.shared_configs import BaseTTSConfig

@dataclass
class VitsAudioConfig(Coqpit):
    fft_size: int = 1024
    sample_rate: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    num_mels: int = 80
    mel_fmin: int = 0
    mel_fmax: int | None = None


@dataclass
class VitsArgs(ModelArgs):
    num_chars: int = 100
    out_channels: int = 513
    spec_segment_size: int = 32
    hidden_channels: int = 192
    hidden_channels_ffn_text_encoder: int = 768
    num_heads_text_encoder: int = 2
    num_layers_text_encoder: int = 6
    kernel_size_text_encoder: int = 3
    dropout_p_text_encoder: float = 0.1
    dropout_p_duration_predictor: float = 0.5
    kernel_size_posterior_encoder: int = 5
    dilation_rate_posterior_encoder: int = 1
    num_layers_posterior_encoder: int = 16
    kernel_size_flow: int = 5
    dilation_rate_flow: int = 1
    num_layers_flow: int = 4
    resblock_type_decoder: str = "1"
    resblock_kernel_sizes_decoder: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes_decoder: list[list[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_rates_decoder: list[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_initial_channel_decoder: int = 512
    upsample_kernel_sizes_decoder: list[int] = field(default_factory=lambda: [16, 16, 4, 4])
    periods_multi_period_discriminator: list[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    use_sdp: bool = True
    noise_scale: float = 1.0
    inference_noise_scale: float = 0.667
    length_scale: float = 1
    noise_scale_dp: float = 1.0
    inference_noise_scale_dp: float = 1.0
    max_inference_len: int = None
    init_discriminator: bool = True
    use_spectral_norm_disriminator: bool = False
    use_speaker_embedding: bool = False
    num_speakers: int = 0
    speakers_file: str | None = None
    d_vector_file: str | list[str] | None = None
    speaker_embedding_channels: int = 256
    use_d_vector_file: bool = False
    d_vector_dim: int = 0
    detach_dp_input: bool = True
    use_language_embedding: bool = False
    embedded_language_dim: int = 4
    language_ids_file: str | None = None
    use_speaker_encoder_as_loss: bool = False
    speaker_encoder_config_path: str = ""
    speaker_encoder_model_path: str = ""
    condition_dp_on_speaker: bool = True
    freeze_encoder: bool = False
    freeze_DP: bool = False
    freeze_PE: bool = False
    freeze_flow_decoder: bool = False
    freeze_waveform_decoder: bool = False
    encoder_sample_rate: int | None = None
    interpolate_z: bool = True
    reinit_DP: bool = False
    reinit_text_encoder: bool = False


@dataclass
class VitsConfig(BaseTTSConfig):
    model: str = "vits"
    model_args: VitsArgs = field(default_factory=VitsArgs)
    audio: VitsAudioConfig = field(default_factory=VitsAudioConfig)
    grad_clip: list[float] = field(default_factory=lambda: [1000, 1000])
    lr_gen: float = 0.0002
    lr_disc: float = 0.0002
    lr_scheduler_gen: str = "ExponentialLR"
    lr_scheduler_gen_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})
    lr_scheduler_disc: str = "ExponentialLR"
    lr_scheduler_disc_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})
    scheduler_after_epoch: bool = True
    optimizer: str = "AdamW"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01})
    kl_loss_alpha: float = 1.0
    disc_loss_alpha: float = 1.0
    gen_loss_alpha: float = 1.0
    feat_loss_alpha: float = 1.0
    mel_loss_alpha: float = 45.0
    dur_loss_alpha: float = 1.0
    speaker_encoder_loss_alpha: float = 1.0
    return_wav: bool = True
    compute_linear_spec: bool = True
    use_weighted_sampler: bool = False 
    weighted_sampler_attrs: dict = field(default_factory=dict)
    weighted_sampler_multipliers: dict = field(default_factory=dict)
    r: int = 1  
    add_blank: bool = True
    num_speakers: int = 0
    use_speaker_embedding: bool = False
    speakers_file: str | None = None
    speaker_embedding_channels: int = 256
    use_d_vector_file: bool = False
    d_vector_file: str | list[str] | None = None
    d_vector_dim: int | None = None

    def __post_init__(self):
        for key, val in self.model_args.items():
            if hasattr(self, key):
                self[key] = val
