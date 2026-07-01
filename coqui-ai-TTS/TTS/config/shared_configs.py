from dataclasses import asdict, dataclass

from coqpit import Coqpit, check_argument
from trainer import TrainerConfig


@dataclass
class BaseAudioConfig(Coqpit):

    fft_size: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    frame_shift_ms: int = None
    frame_length_ms: int = None
    stft_pad_mode: str = "reflect"
    sample_rate: int = 22050
    resample: bool = False
    preemphasis: float = 0.0
    ref_level_db: int = 20
    do_sound_norm: bool = False
    log_func: str = "np.log10"
    do_trim_silence: bool = True
    trim_db: int = 45
    do_rms_norm: bool = False
    db_level: float = None
    power: float = 1.5
    griffin_lim_iters: int = 60
    num_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = None
    spec_gain: int = 20
    do_amp_to_db_linear: bool = True
    do_amp_to_db_mel: bool = True
    pitch_fmax: float = 640.0
    pitch_fmin: float = 1.0
    signal_norm: bool = True
    min_level_db: int = -100
    symmetric_norm: bool = True
    max_norm: float = 4.0
    clip_norm: bool = True
    stats_path: str = None

    def check_values(
        self,
    ):
        c = asdict(self)
        check_argument("num_mels", c, restricted=True, min_val=10, max_val=2056)
        check_argument("fft_size", c, restricted=True, min_val=128, max_val=4058)
        check_argument("sample_rate", c, restricted=True, min_val=512, max_val=100000)
        check_argument(
            "frame_length_ms",
            c,
            restricted=True,
            min_val=10,
            max_val=1000,
            alternative="win_length",
        )
        check_argument("frame_shift_ms", c, restricted=True, min_val=1, max_val=1000, alternative="hop_length")
        check_argument("preemphasis", c, restricted=True, min_val=0, max_val=1)
        check_argument("min_level_db", c, restricted=True, min_val=-1000, max_val=10)
        check_argument("ref_level_db", c, restricted=True, min_val=0, max_val=1000)
        check_argument("power", c, restricted=True, min_val=1, max_val=5)
        check_argument("griffin_lim_iters", c, restricted=True, min_val=10, max_val=1000)
        check_argument("signal_norm", c, restricted=True)
        check_argument("symmetric_norm", c, restricted=True)
        check_argument("max_norm", c, restricted=True, min_val=0.1, max_val=1000)
        check_argument("clip_norm", c, restricted=True)
        check_argument("mel_fmin", c, restricted=True, min_val=0.0, max_val=1000)
        check_argument("mel_fmax", c, restricted=True, min_val=500.0, allow_none=True)
        check_argument("spec_gain", c, restricted=True, min_val=1, max_val=100)
        check_argument("do_trim_silence", c, restricted=True)
        check_argument("trim_db", c, restricted=True)


@dataclass
class BaseDatasetConfig(Coqpit):
    formatter: str | None = ""
    dataset_name: str | None = ""
    path: str | None = ""
    meta_file_train: str = ""
    ignored_speakers: list[str] | None = None
    language: str = ""
    phonemizer: str = ""
    meta_file_val: str = ""
    meta_file_attn_mask: str = ""

    def check_values(
        self,
    ):
        c = asdict(self)
        check_argument("formatter", c, restricted=True)
        check_argument("path", c, restricted=True)
        check_argument("meta_file_train", c, restricted=True)
        check_argument("meta_file_val", c, restricted=False)
        check_argument("meta_file_attn_mask", c, restricted=False)


@dataclass
class BaseTrainingConfig(TrainerConfig):
    model: str = None
    num_loader_workers: int = 0
    num_eval_loader_workers: int = 0
    use_noise_augment: bool = False


@dataclass
class ModelArgs(Coqpit):
    """Parameters necessary for model instantiation."""
