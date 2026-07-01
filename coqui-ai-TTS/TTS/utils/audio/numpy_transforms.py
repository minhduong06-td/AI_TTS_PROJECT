import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any
import librosa
import numpy as np
import scipy
import soundfile as sf
from librosa import effects, filters, magphase, pyin
logger = logging.getLogger(__name__)
def build_mel_basis(
    *,
    sample_rate: int,
    fft_size: int,
    num_mels: int,
    mel_fmin: int,
    mel_fmax: int | None = None,
    **kwargs,
) -> np.ndarray:
    if mel_fmax is not None:
        assert mel_fmax <= sample_rate // 2
        assert mel_fmax - mel_fmin > 0
    return filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=mel_fmin, fmax=mel_fmax)


def millisec_to_length(*, frame_length_ms: float, frame_shift_ms: float, sample_rate: int, **kwargs) -> tuple[int, int]:
    factor = frame_length_ms / frame_shift_ms
    assert (factor).is_integer(), " [!] frame_shift_ms should divide frame_length_ms"
    win_length = int(frame_length_ms / 1000.0 * sample_rate)
    hop_length = int(win_length / float(factor))
    return win_length, hop_length


def _log(x, base):
    if base == 10:
        return np.log10(x)
    return np.log(x)


def _exp(x, base):
    if base == 10:
        return np.power(10, x)
    return np.exp(x)


def amp_to_db(*, x: np.ndarray, gain: float = 1, base: float = 10, **kwargs) -> np.ndarray:
    assert (x < 0).sum() == 0, " [!] Input values must be non-negative."
    return gain * _log(np.maximum(1e-8, x), base)

def db_to_amp(*, x: np.ndarray, gain: float = 1, base: float = 10, **kwargs) -> np.ndarray:
    return _exp(x / gain, base)


def preemphasis(*, x: np.ndarray, coef: float = 0.97, **kwargs) -> np.ndarray:
    if coef == 0:
        msg = "Preemphasis is set 0.0."
        raise RuntimeError(msg)
    return scipy.signal.lfilter([1, -coef], [1], x)


def deemphasis(*, x: np.ndarray, coef: float = 0.97, **kwargs) -> np.ndarray:
    if coef == 0:
        msg = "Deemphasis is set 0.0."
        raise ValueError(msg)
    return scipy.signal.lfilter([1], [1, -coef], x)


def spec_to_mel(*, spec: np.ndarray, mel_basis: np.ndarray, **kwargs) -> np.ndarray:
    return np.dot(mel_basis, spec)


def mel_to_spec(*, mel: np.ndarray, mel_basis: np.ndarray, **kwargs) -> np.ndarray:
    assert (mel < 0).sum() == 0, "Input values must be non-negative."
    inv_mel_basis = np.linalg.pinv(mel_basis)
    return np.maximum(1e-10, np.dot(inv_mel_basis, mel))


def wav_to_spec(*, wav: np.ndarray, **kwargs) -> np.ndarray:
    D = stft(y=wav, **kwargs)
    S = np.abs(D)
    return S.astype(np.float32)


def wav_to_mel(*, wav: np.ndarray, mel_basis: np.ndarray, **kwargs) -> np.ndarray:
    D = stft(y=wav, **kwargs)
    S = spec_to_mel(spec=np.abs(D), mel_basis=mel_basis, **kwargs)
    return S.astype(np.float32)


def spec_to_wav(*, spec: np.ndarray, power: float = 1.5, **kwargs) -> np.ndarray:
    S = spec.copy()
    return griffin_lim(spec=S**power, **kwargs)


def mel_to_wav(*, mel: np.ndarray, mel_basis: np.ndarray, power: float = 1.5, **kwargs) -> np.ndarray:
    S = mel.copy()
    S = mel_to_spec(mel=S, mel_basis=mel_basis)  
    return griffin_lim(spec=S**power, **kwargs)

def stft(
    *,
    y: np.ndarray,
    fft_size: int,
    hop_length: int | None = None,
    win_length: int | None = None,
    pad_mode: str = "reflect",
    window: str = "hann",
    center: bool = True,
    **kwargs,
) -> np.ndarray:
    return librosa.stft(
        y=y,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=win_length,
        pad_mode=pad_mode,
        window=window,
        center=center,
    )


def istft(
    *,
    y: np.ndarray,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str = "hann",
    center: bool = True,
    **kwargs,
) -> np.ndarray:
    return librosa.istft(y, hop_length=hop_length, win_length=win_length, center=center, window=window)


def griffin_lim(*, spec: np.ndarray, num_iter=60, **kwargs) -> np.ndarray:
    angles = np.exp(2j * np.pi * np.random.rand(*spec.shape))
    S_complex = np.abs(spec).astype(complex)
    y = istft(y=S_complex * angles, **kwargs)
    if not np.isfinite(y).all():
        logger.warning("Waveform is not finite everywhere. Skipping the GL.")
        return np.array([0.0])
    for _ in range(num_iter):
        angles = np.exp(1j * np.angle(stft(y=y, **kwargs)))
        y = istft(y=S_complex * angles, **kwargs)
    return y


def compute_stft_paddings(*, x: np.ndarray, hop_length: int, pad_two_sides: bool = False, **kwargs) -> tuple[int, int]:
    pad = (x.shape[0] // hop_length + 1) * hop_length - x.shape[0]
    if not pad_two_sides:
        return 0, pad
    return pad // 2, pad // 2 + pad % 2


def compute_f0(
    *,
    x: np.ndarray,
    pitch_fmax: float | None = None,
    pitch_fmin: float | None = None,
    hop_length: int,
    win_length: int,
    sample_rate: int,
    stft_pad_mode: str = "reflect",
    center: bool = True,
    **kwargs,
) -> np.ndarray:
    assert pitch_fmax is not None, " [!] Set `pitch_fmax` before calling `compute_f0`."
    assert pitch_fmin is not None, " [!] Set `pitch_fmin` before calling `compute_f0`."

    if sample_rate / pitch_fmin >= win_length - 1:
        logger.warning("pitch_fmin=%.2f is too small for win_length=%d", pitch_fmin, win_length)
        pitch_fmin = sample_rate / (win_length - 1) + 0.1
        logger.warning("pitch_fmin increased to %f", pitch_fmin)

    f0, voiced_mask, _ = pyin(
        y=x.astype(np.double),
        fmin=pitch_fmin,
        fmax=pitch_fmax,
        sr=sample_rate,
        frame_length=win_length,
        hop_length=hop_length,
        pad_mode=stft_pad_mode,
        center=center,
        n_thresholds=100,
        beta_parameters=(2, 18),
        boltzmann_parameter=2,
        resolution=0.1,
        max_transition_rate=35.92,
        switch_prob=0.01,
        no_trough_prob=0.01,
    )
    f0[~voiced_mask] = 0.0

    return f0


def compute_energy(y: np.ndarray, **kwargs) -> np.ndarray:
    x = stft(y=y, **kwargs)
    mag, _ = magphase(x)
    return np.sqrt(np.sum(mag**2, axis=0))


### Audio Processing ###
def find_endpoint(
    *,
    wav: np.ndarray,
    trim_db: float = -40,
    sample_rate: int,
    min_silence_sec: float = 0.8,
    gain: float = 1,
    base: float = 10,
    **kwargs,
) -> int:
    window_length = int(sample_rate * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = db_to_amp(x=-trim_db, gain=gain, base=base)
    for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x : x + window_length]) < threshold:
            return x + hop_length
    return len(wav)


def trim_silence(
    *,
    wav: np.ndarray,
    sample_rate: int,
    trim_db: float = 60,
    win_length: int,
    hop_length: int,
    **kwargs,
) -> np.ndarray:
    margin = int(sample_rate * 0.01)
    wav = wav[margin:-margin]
    return effects.trim(wav, top_db=trim_db, frame_length=win_length, hop_length=hop_length)[0]


def volume_norm(*, x: np.ndarray, coef: float = 0.95, **kwargs) -> np.ndarray:
    return x / abs(x).max() * coef


def rms_norm(*, wav: np.ndarray, db_level: float = -27.0, **kwargs) -> np.ndarray:
    r = 10 ** (db_level / 20)
    a = np.sqrt((len(wav) * (r**2)) / np.sum(wav**2))
    return wav * a


def rms_volume_norm(*, x: np.ndarray, db_level: float = -27.0, **kwargs) -> np.ndarray:
    assert -99 <= db_level <= 0, "db_level should be between -99 and 0"
    return rms_norm(wav=x, db_level=db_level)


def load_wav(
    *, filename: str | os.PathLike[Any], sample_rate: int | None = None, resample: bool = False, **kwargs
) -> np.ndarray:
    if resample:
        x, _ = librosa.load(filename, sr=sample_rate)
    else:
        x, _ = sf.read(filename)
    if x.ndim != 1:
        logger.warning("Found multi-channel audio. Converting to mono: %s", filename)
        x = librosa.to_mono(x)
    return x


def save_wav(
    *,
    wav: np.ndarray,
    path: str | os.PathLike[Any] | BytesIO,
    sample_rate: int,
    pipe_out=None,
    do_rms_norm: bool = False,
    db_level: float = -27.0,
    **kwargs,
) -> None:
    if not isinstance(path, BytesIO):
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        if path.is_dir():
            msg = f"Output path must be a file, not a directory: {path}"
            raise IsADirectoryError(msg)
    if do_rms_norm:
        if db_level is None:
            msg = "`db_level` cannot be None with `do_rms_norm=True`"
            raise ValueError(msg)
        wav_norm = rms_volume_norm(x=wav, db_level=db_level)
    else:
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

    wav_norm = wav_norm.astype(np.int16)
    if pipe_out:
        wav_buffer = BytesIO()
        scipy.io.wavfile.write(wav_buffer, sample_rate, wav_norm)
        wav_buffer.seek(0)
        pipe_out.buffer.write(wav_buffer.read())
    scipy.io.wavfile.write(path, sample_rate, wav_norm)


def mulaw_encode(*, wav: np.ndarray, mulaw_qc: int, **kwargs) -> np.ndarray:
    mu = 2**mulaw_qc - 1
    signal = np.sign(wav) * np.log(1 + mu * np.abs(wav)) / np.log(1.0 + mu)
    signal = (signal + 1) / 2 * mu + 0.5
    return np.floor(
        signal,
    )


def mulaw_decode(*, wav, mulaw_qc: int, **kwargs) -> np.ndarray:
    mu = 2**mulaw_qc - 1
    return np.sign(wav) / mu * ((1 + mu) ** np.abs(wav) - 1)


def encode_16bits(*, x: np.ndarray, **kwargs) -> np.ndarray:
    return np.clip(x * 2**15, -(2**15), 2**15 - 1).astype(np.int16)


def quantize(*, x: np.ndarray, quantize_bits: int, **kwargs) -> np.ndarray:
    return (x + 1.0) * (2**quantize_bits - 1) / 2


def dequantize(*, x, quantize_bits, **kwargs) -> np.ndarray:
    return 2 * x / (2**quantize_bits - 1) - 1
