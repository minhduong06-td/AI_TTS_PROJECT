import logging
import os
from typing import Any
import librosa
import numpy as np
from TTS.tts.utils.helpers import StandardScaler
from TTS.utils.audio.numpy_transforms import (
    amp_to_db,
    build_mel_basis,
    compute_f0,
    db_to_amp,
    deemphasis,
    find_endpoint,
    griffin_lim,
    load_wav,
    mel_to_spec,
    millisec_to_length,
    preemphasis,
    rms_volume_norm,
    save_wav,
    spec_to_mel,
    stft,
    trim_silence,
    volume_norm,
)
logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(
        self,
        sample_rate=None,
        resample=False,
        num_mels=None,
        log_func="np.log10",
        min_level_db=None,
        frame_shift_ms=None,
        frame_length_ms=None,
        hop_length=None,
        win_length=None,
        ref_level_db=None,
        fft_size=1024,
        power=None,
        preemphasis=0.0,
        signal_norm=None,
        symmetric_norm=None,
        max_norm=None,
        mel_fmin=None,
        mel_fmax=None,
        pitch_fmax=None,
        pitch_fmin=None,
        spec_gain=20,
        stft_pad_mode="reflect",
        clip_norm=True,
        griffin_lim_iters=None,
        do_trim_silence=False,
        trim_db=60,
        do_sound_norm=False,
        do_amp_to_db_linear=True,
        do_amp_to_db_mel=True,
        do_rms_norm=False,
        db_level=None,
        stats_path=None,
        **_,
    ) -> None:
  
        self.sample_rate = sample_rate
        self.resample = resample
        self.num_mels = num_mels
        self.log_func = log_func
        self.min_level_db = min_level_db or 0
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.ref_level_db = ref_level_db
        self.fft_size = fft_size
        self.power = power
        self.preemphasis = preemphasis
        self.griffin_lim_iters = griffin_lim_iters
        self.signal_norm = signal_norm
        self.symmetric_norm = symmetric_norm
        self.mel_fmin = mel_fmin or 0
        self.mel_fmax = mel_fmax
        self.pitch_fmin = pitch_fmin
        self.pitch_fmax = pitch_fmax
        self.spec_gain = float(spec_gain)
        self.stft_pad_mode = stft_pad_mode
        self.max_norm = 1.0 if max_norm is None else float(max_norm)
        self.clip_norm = clip_norm
        self.do_trim_silence = do_trim_silence
        self.trim_db = trim_db
        self.do_sound_norm = do_sound_norm
        self.do_amp_to_db_linear = do_amp_to_db_linear
        self.do_amp_to_db_mel = do_amp_to_db_mel
        self.do_rms_norm = do_rms_norm
        self.db_level = db_level
        self.stats_path = stats_path
       
        if log_func == "np.log":
            self.base = np.e
        elif log_func == "np.log10":
            self.base = 10
        else:
            msg = "unknown `log_func` value."
            raise ValueError(msg)
   
        if hop_length is None:
            self.win_length, self.hop_length = millisec_to_length(
                frame_length_ms=self.frame_length_ms, frame_shift_ms=self.frame_shift_ms, sample_rate=self.sample_rate
            )
        else:
            self.hop_length = hop_length
            self.win_length = win_length
        assert min_level_db != 0.0, "min_level_db is 0"
        assert self.win_length <= self.fft_size, (
            f"win_length cannot be larger than fft_size - {self.win_length} vs {self.fft_size}"
        )
        members = vars(self)
        logger.info("Setting up Audio Processor...")
        for key, value in members.items():
            logger.info(" | %s: %s", key, value)
        # create spectrogram utils
        self.mel_basis = build_mel_basis(
            sample_rate=self.sample_rate,
            fft_size=self.fft_size,
            num_mels=self.num_mels,
            mel_fmax=self.mel_fmax,
            mel_fmin=self.mel_fmin,
        )
        # setup scaler
        if stats_path and signal_norm:
            mel_mean, mel_std, linear_mean, linear_std, _ = self.load_stats(stats_path)
            self.setup_scaler(mel_mean, mel_std, linear_mean, linear_std)
            self.signal_norm = True
            self.max_norm = None
            self.clip_norm = None
            self.symmetric_norm = None

    @staticmethod
    def init_from_config(config: "Coqpit"):
        if "audio" in config:
            return AudioProcessor(**config.audio)
        return AudioProcessor(**config)
    def normalize(self, S: np.ndarray) -> np.ndarray:
       
        S = S.copy()
        if self.signal_norm:
            if hasattr(self, "mel_scaler"):
                if S.shape[0] == self.num_mels:
                    return self.mel_scaler.transform(S.T).T
                if S.shape[0] == self.fft_size / 2:
                    return self.linear_scaler.transform(S.T).T
                msg = " [!] Mean-Var stats does not match the given feature dimensions."
                raise RuntimeError(msg)
           
            S -= self.ref_level_db  # discard certain range of DB assuming it is air noise
            S_norm = (S - self.min_level_db) / (-self.min_level_db)
            if self.symmetric_norm:
                S_norm = ((2 * self.max_norm) * S_norm) - self.max_norm
                if self.clip_norm:
                    S_norm = np.clip(
                        S_norm,
                        -self.max_norm,  # pylint: disable=invalid-unary-operand-type
                        self.max_norm,
                    )
                return S_norm
            S_norm = self.max_norm * S_norm
            if self.clip_norm:
                S_norm = np.clip(S_norm, 0, self.max_norm)
            return S_norm
        return S

    def denormalize(self, S: np.ndarray) -> np.ndarray:
        S_denorm = S.copy()
        if self.signal_norm:
    
            if hasattr(self, "mel_scaler"):
                if S_denorm.shape[0] == self.num_mels:
                    return self.mel_scaler.inverse_transform(S_denorm.T).T
                if S_denorm.shape[0] == self.fft_size / 2:
                    return self.linear_scaler.inverse_transform(S_denorm.T).T
                msg = "Mean-Var stats does not match the given feature dimensions."
                raise RuntimeError(msg)
            if self.symmetric_norm:
                if self.clip_norm:
                    S_denorm = np.clip(
                        S_denorm,
                        -self.max_norm, 
                        self.max_norm,
                    )
                S_denorm = ((S_denorm + self.max_norm) * -self.min_level_db / (2 * self.max_norm)) + self.min_level_db
                return S_denorm + self.ref_level_db
            if self.clip_norm:
                S_denorm = np.clip(S_denorm, 0, self.max_norm)
            S_denorm = (S_denorm * -self.min_level_db / self.max_norm) + self.min_level_db
            return S_denorm + self.ref_level_db
        return S_denorm

    def load_stats(self, stats_path: str) -> tuple[np.array, np.array, np.array, np.array, dict]:
        stats = np.load(stats_path, allow_pickle=True).item() 
        mel_mean = stats["mel_mean"]
        mel_std = stats["mel_std"]
        linear_mean = stats["linear_mean"]
        linear_std = stats["linear_std"]
        stats_config = stats["audio_config"]
        skip_parameters = ["griffin_lim_iters", "stats_path", "do_trim_silence", "ref_level_db", "power"]
        for key in stats_config:
            if key in skip_parameters:
                continue
            if key not in ["sample_rate", "trim_db"]:
                assert stats_config[key] == self.__dict__[key], (
                    f" [!] Audio param {key} does not match the value used for computing mean-var stats. {stats_config[key]} vs {self.__dict__[key]}"
                )
        return mel_mean, mel_std, linear_mean, linear_std, stats_config
    def setup_scaler(
        self, mel_mean: np.ndarray, mel_std: np.ndarray, linear_mean: np.ndarray, linear_std: np.ndarray
    ) -> None:
        self.mel_scaler = StandardScaler()
        self.mel_scaler.set_stats(mel_mean, mel_std)
        self.linear_scaler = StandardScaler()
        self.linear_scaler.set_stats(linear_mean, linear_std)

    def apply_preemphasis(self, x: np.ndarray) -> np.ndarray:
        return preemphasis(x=x, coef=self.preemphasis)

    def apply_inv_preemphasis(self, x: np.ndarray) -> np.ndarray:
        """Reverse pre-emphasis."""
        return deemphasis(x=x, coef=self.preemphasis)

    ### SPECTROGRAMs ###
    def spectrogram(self, y: np.ndarray) -> np.ndarray:
        if self.preemphasis != 0:
            y = self.apply_preemphasis(y)
        D = stft(
            y=y,
            fft_size=self.fft_size,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad_mode=self.stft_pad_mode,
        )
        S = amp_to_db(x=np.abs(D), gain=self.spec_gain, base=self.base) if self.do_amp_to_db_linear else np.abs(D)
        return self.normalize(S).astype(np.float32)

    def melspectrogram(self, y: np.ndarray) -> np.ndarray:
        if self.preemphasis != 0:
            y = self.apply_preemphasis(y)
        D = stft(
            y=y,
            fft_size=self.fft_size,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad_mode=self.stft_pad_mode,
        )
        S = spec_to_mel(spec=np.abs(D), mel_basis=self.mel_basis)
        if self.do_amp_to_db_mel:
            S = amp_to_db(x=S, gain=self.spec_gain, base=self.base)

        return self.normalize(S).astype(np.float32)

    def inv_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        S = self.denormalize(spectrogram)
        S = db_to_amp(x=S, gain=self.spec_gain, base=self.base)
        W = self._griffin_lim(S**self.power)
        return self.apply_inv_preemphasis(W) if self.preemphasis != 0 else W

    def inv_melspectrogram(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Convert a melspectrogram to a waveform using Griffi-Lim vocoder."""
        D = self.denormalize(mel_spectrogram)
        S = db_to_amp(x=D, gain=self.spec_gain, base=self.base)
        S = mel_to_spec(mel=S, mel_basis=self.mel_basis)  
        W = self._griffin_lim(S**self.power)
        return self.apply_inv_preemphasis(W) if self.preemphasis != 0 else W

    def out_linear_to_mel(self, linear_spec: np.ndarray) -> np.ndarray:
        S = self.denormalize(linear_spec)
        S = db_to_amp(x=S, gain=self.spec_gain, base=self.base)
        S = spec_to_mel(spec=np.abs(S), mel_basis=self.mel_basis)
        S = amp_to_db(x=S, gain=self.spec_gain, base=self.base)
        return self.normalize(S)

    def _griffin_lim(self, S):
        return griffin_lim(
            spec=S,
            num_iter=self.griffin_lim_iters,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fft_size=self.fft_size,
            pad_mode=self.stft_pad_mode,
        )

    def compute_f0(self, x: np.ndarray) -> np.ndarray:
        if len(x) % self.hop_length == 0:
            x = np.pad(x, (0, self.hop_length // 2), mode=self.stft_pad_mode)

        return compute_f0(
            x=x,
            pitch_fmax=self.pitch_fmax,
            pitch_fmin=self.pitch_fmin,
            hop_length=self.hop_length,
            win_length=self.win_length,
            sample_rate=self.sample_rate,
            stft_pad_mode=self.stft_pad_mode,
            center=True,
        )
    def find_endpoint(self, wav: np.ndarray, min_silence_sec=0.8) -> int:
        return find_endpoint(
            wav=wav,
            trim_db=self.trim_db,
            sample_rate=self.sample_rate,
            min_silence_sec=min_silence_sec,
            gain=self.spec_gain,
            base=self.base,
        )

    def trim_silence(self, wav):
        return trim_silence(
            wav=wav,
            sample_rate=self.sample_rate,
            trim_db=self.trim_db,
            win_length=self.win_length,
            hop_length=self.hop_length,
        )

    @staticmethod
    def sound_norm(x: np.ndarray) -> np.ndarray:
        return volume_norm(x=x)

    def load_wav(self, filename: str | os.PathLike[Any], sr: int | None = None) -> np.ndarray:
        if sr is not None:
            x = load_wav(filename=filename, sample_rate=sr, resample=True)
        else:
            x = load_wav(filename=filename, sample_rate=self.sample_rate, resample=self.resample)
        if self.do_trim_silence:
            try:
                x = self.trim_silence(x)
            except ValueError:
                logger.exception("File cannot be trimmed for silence - %s", filename)
        if self.do_sound_norm:
            x = self.sound_norm(x)
        if self.do_rms_norm:
            x = rms_volume_norm(x=x, db_level=self.db_level)
        return x

    def save_wav(self, wav: np.ndarray, path: str | os.PathLike[Any], sr: int | None = None, pipe_out=None) -> None:
        save_wav(
            wav=wav,
            path=path,
            sample_rate=sr if sr else self.sample_rate,
            pipe_out=pipe_out,
            do_rms_norm=self.do_rms_norm,
            db_level=self.db_level,
        )

    def get_duration(self, filename: str) -> float:
        return librosa.get_duration(filename=filename)
