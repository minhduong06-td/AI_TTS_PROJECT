import argparse
import os
from pathlib import Path
from typing import Optional
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsAudioConfig, VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

COMMON_USE_PHONEMES = True

COMMON_FFT_SIZE = 1024
COMMON_WIN_LENGTH = 1024
COMMON_HOP_LENGTH = 256
COMMON_NUM_MELS = 80
COMMON_MEL_FMIN = 0.0
COMMON_MEL_FMAX = None

COMMON_BATCH_SIZE = 4
COMMON_EVAL_BATCH_SIZE = 2
COMMON_BATCH_GROUP_SIZE = 3

COMMON_NUM_WORKERS = 2
COMMON_NUM_EVAL_WORKERS = 1

COMMON_EPOCHS = 2000
COMMON_PRINT_STEP = 25
COMMON_MIXED_PRECISION = True

COMMON_SAVE_STEP = 3000
COMMON_SAVE_N_CHECKPOINTS = 8
COMMON_SAVE_ALL_BEST = False
COMMON_SAVE_BEST_AFTER = 3000
COMMON_TEST_DELAY_EPOCHS = -1
COMMON_LR_GEN = 7.5e-5
COMMON_LR_DISC = 7.5e-5
COMMON_LR_GAMMA = 0.999875

COMMON_GRAD_CLIP = [1000.0, 1000.0]
COMMON_OPTIMIZER = "AdamW"
COMMON_OPTIMIZER_PARAMS = {
    "betas": [0.8, 0.99],
    "eps": 1.0e-9,
    "weight_decay": 0.01,
}

COMMON_KL_LOSS_ALPHA = 1.0
COMMON_DISC_LOSS_ALPHA = 1.0
COMMON_GEN_LOSS_ALPHA = 1.0
COMMON_FEAT_LOSS_ALPHA = 1.0
COMMON_MEL_LOSS_ALPHA = 45.0
COMMON_DUR_LOSS_ALPHA = 1.0
COMMON_TARGET_LOSS = "loss_1"

COMMON_CUDNN_BENCHMARK = False
COMMON_EVAL_SPLIT_SIZE = 0.02
COMMON_EVAL_SPLIT_MAX_SIZE = 128
COMMON_MIN_AUDIO_SECONDS = 1
COMMON_MAX_AUDIO_SECONDS = 12
COMMON_MIN_TEXT_LEN = 4
COMMON_MAX_TEXT_LEN = 240
COMMON_INFERENCE_NOISE_SCALE = 0.45
COMMON_INFERENCE_NOISE_SCALE_DP = 0.65
COMMON_LENGTH_SCALE = 1.0


def parse_resume_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--continue_path",
        type=str,
        default=None,
        help="Resume the same run with its original config and optimizer state.",
    )
    parser.add_argument(
        "--restore_path",
        type=str,
        default=None,
        help="Fine-tune from a checkpoint in a new run with the optimized config.",
    )
    parser.add_argument(
        "--restore_config",
        type=str,
        default=None,
        help="Optional config.json paired with --restore_path.",
    )
    parser.add_argument("--epochs", type=int, default=COMMON_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=COMMON_BATCH_SIZE)
    parser.add_argument("--eval_batch_size", type=int, default=COMMON_EVAL_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=COMMON_NUM_WORKERS)
    parser.add_argument("--num_eval_workers", type=int, default=COMMON_NUM_EVAL_WORKERS)
    parser.add_argument("--lr_gen", type=float, default=COMMON_LR_GEN)
    parser.add_argument("--lr_disc", type=float, default=COMMON_LR_DISC)
    args = parser.parse_args()

    if args.continue_path and args.restore_path:
        raise ValueError("Use only one of --continue_path or --restore_path.")
    if args.restore_config and not args.restore_path:
        raise ValueError("--restore_config requires --restore_path.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than 0.")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Batch sizes must be greater than 0.")
    if args.lr_gen <= 0 or args.lr_disc <= 0:
        raise ValueError("Learning rates must be greater than 0.")

    return args


def build_dataset_config(
    *,
    dataset_name: str,
    dataset_path: str,
    meta_file: str,
    language: str,
) -> BaseDatasetConfig:
    return BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name=dataset_name,
        path=dataset_path,
        meta_file_train=meta_file,
        language=language,
    )


def build_audio_config(sample_rate: int) -> VitsAudioConfig:
    return VitsAudioConfig(
        fft_size=COMMON_FFT_SIZE,
        sample_rate=sample_rate,
        win_length=COMMON_WIN_LENGTH,
        hop_length=COMMON_HOP_LENGTH,
        num_mels=COMMON_NUM_MELS,
        mel_fmin=COMMON_MEL_FMIN,
        mel_fmax=COMMON_MEL_FMAX,
    )


def _find_reference_config(
    *,
    continue_path: Optional[str],
    restore_path: Optional[str],
    restore_config: Optional[str],
) -> Optional[Path]:
    if continue_path:
        config_path = Path(continue_path).expanduser().resolve() / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing continue config: {config_path}")
        return config_path

    if not restore_path:
        return None

    if restore_config:
        config_path = Path(restore_config).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing restore config: {config_path}")
        return config_path

    checkpoint_path = Path(restore_path).expanduser().resolve()
    candidates = [
        checkpoint_path.parent / "config.json",
        checkpoint_path.parent.parent / "config.json",
    ]
    for config_path in candidates:
        if config_path.is_file():
            return config_path

    raise FileNotFoundError(
        "Cannot locate config.json for the checkpoint. "
        "Pass it explicitly with --restore_config."
    )


def _load_vits_config(config_path: Path) -> VitsConfig:
    config = VitsConfig()
    config.load_json(str(config_path))
    return config


def _assert_same(name: str, old_value, new_value) -> None:
    if old_value != new_value:
        raise ValueError(
            f"Restore incompatibility for {name}: "
            f"checkpoint={old_value!r}, requested={new_value!r}"
        )


def _validate_restore_compatibility(
    *,
    config: VitsConfig,
    audio_config: VitsAudioConfig,
    text_cleaner: str,
    phonemizer: str,
    phoneme_language: str,
) -> None:
    for field in (
        "fft_size",
        "sample_rate",
        "win_length",
        "hop_length",
        "num_mels",
        "mel_fmin",
        "mel_fmax",
    ):
        _assert_same(
            f"audio.{field}",
            getattr(config.audio, field),
            getattr(audio_config, field),
        )

    _assert_same("use_phonemes", config.use_phonemes, COMMON_USE_PHONEMES)
    _assert_same("text_cleaner", config.text_cleaner, text_cleaner)
    _assert_same("phonemizer", config.phonemizer, phonemizer)
    _assert_same("phoneme_language", config.phoneme_language, phoneme_language)


def _set_inference_defaults(config: VitsConfig) -> None:
    values = {
        "inference_noise_scale": COMMON_INFERENCE_NOISE_SCALE,
        "inference_noise_scale_dp": COMMON_INFERENCE_NOISE_SCALE_DP,
        "length_scale": COMMON_LENGTH_SCALE,
    }
    for name, value in values.items():
        if hasattr(config.model_args, name):
            setattr(config.model_args, name, value)
        if hasattr(config, name):
            setattr(config, name, value)


def _apply_finetune_profile(
    *,
    config: VitsConfig,
    run_name: str,
    output_path: str,
    dataset_config: BaseDatasetConfig,
    audio_config: VitsAudioConfig,
    text_cleaner: str,
    phonemizer: str,
    phoneme_language: str,
    test_sentences: list,
    characters,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    num_eval_workers: int,
    lr_gen: float,
    lr_disc: float,
) -> VitsConfig:
    config.audio = audio_config
    config.run_name = run_name
    config.output_path = output_path
    config.datasets = [dataset_config]

    config.batch_size = batch_size
    config.eval_batch_size = eval_batch_size
    config.batch_group_size = COMMON_BATCH_GROUP_SIZE
    config.num_loader_workers = num_workers
    config.num_eval_loader_workers = num_eval_workers

    config.run_eval = True
    config.eval_split_size = COMMON_EVAL_SPLIT_SIZE
    config.eval_split_max_size = COMMON_EVAL_SPLIT_MAX_SIZE

    # Filter pathological long/short samples to keep VRAM predictable on 6GB GPUs.
    config.min_audio_len = int(config.audio.sample_rate * COMMON_MIN_AUDIO_SECONDS)
    config.max_audio_len = int(config.audio.sample_rate * COMMON_MAX_AUDIO_SECONDS)
    config.min_text_len = COMMON_MIN_TEXT_LEN
    config.max_text_len = COMMON_MAX_TEXT_LEN

    config.save_step = COMMON_SAVE_STEP
    config.save_n_checkpoints = COMMON_SAVE_N_CHECKPOINTS
    config.save_all_best = COMMON_SAVE_ALL_BEST
    config.save_best_after = COMMON_SAVE_BEST_AFTER
    config.test_delay_epochs = COMMON_TEST_DELAY_EPOCHS

    config.epochs = epochs
    config.print_step = COMMON_PRINT_STEP
    config.print_eval = True
    config.mixed_precision = COMMON_MIXED_PRECISION
    if hasattr(config, "precision"):
        config.precision = "fp16"

    config.text_cleaner = text_cleaner
    config.use_phonemes = COMMON_USE_PHONEMES
    config.phonemizer = phonemizer
    config.phoneme_language = phoneme_language
    if characters is not None:
        config.characters = characters
    config.test_sentences = test_sentences
    config.phoneme_cache_path = os.path.join(output_path, "phoneme_cache", dataset_config.dataset_name)
    config.compute_input_seq_cache = True

    config.grad_clip = list(COMMON_GRAD_CLIP)
    config.optimizer = COMMON_OPTIMIZER
    config.optimizer_params = dict(COMMON_OPTIMIZER_PARAMS)
    config.lr_gen = lr_gen
    config.lr_disc = lr_disc
    config.lr_scheduler_gen = "ExponentialLR"
    config.lr_scheduler_disc = "ExponentialLR"
    config.lr_scheduler_gen_params = {
        "gamma": COMMON_LR_GAMMA,
        "last_epoch": -1,
    }
    config.lr_scheduler_disc_params = {
        "gamma": COMMON_LR_GAMMA,
        "last_epoch": -1,
    }
    config.scheduler_after_epoch = True

    config.kl_loss_alpha = COMMON_KL_LOSS_ALPHA
    config.disc_loss_alpha = COMMON_DISC_LOSS_ALPHA
    config.gen_loss_alpha = COMMON_GEN_LOSS_ALPHA
    config.feat_loss_alpha = COMMON_FEAT_LOSS_ALPHA
    config.mel_loss_alpha = COMMON_MEL_LOSS_ALPHA
    config.dur_loss_alpha = COMMON_DUR_LOSS_ALPHA

    config.target_loss = COMMON_TARGET_LOSS
    config.cudnn_benchmark = COMMON_CUDNN_BENCHMARK

    _set_inference_defaults(config)
    return config


def build_vits_config(
    *,
    run_name: str,
    output_path: str,
    dataset_config: BaseDatasetConfig,
    audio_config: VitsAudioConfig,
    text_cleaner: str,
    phonemizer: str,
    phoneme_language: str,
    test_sentences: Optional[list] = None,
    characters=None,
    continue_path: Optional[str] = None,
    restore_path: Optional[str] = None,
    restore_config: Optional[str] = None,
    epochs: int = COMMON_EPOCHS,
    batch_size: int = COMMON_BATCH_SIZE,
    eval_batch_size: int = COMMON_EVAL_BATCH_SIZE,
    num_workers: int = COMMON_NUM_WORKERS,
    num_eval_workers: int = COMMON_NUM_EVAL_WORKERS,
    lr_gen: float = COMMON_LR_GEN,
    lr_disc: float = COMMON_LR_DISC,
) -> VitsConfig:
    reference_config_path = _find_reference_config(
        continue_path=continue_path,
        restore_path=restore_path,
        restore_config=restore_config,
    )

    if continue_path:
        config = _load_vits_config(reference_config_path)
        print(f"[config] Exact resume config: {reference_config_path}")
        return config

    if restore_path:
        config = _load_vits_config(reference_config_path)
        _validate_restore_compatibility(
            config=config,
            audio_config=audio_config,
            text_cleaner=text_cleaner,
            phonemizer=phonemizer,
            phoneme_language=phoneme_language,
        )
        print(f"[config] Restored architecture/tokenizer config: {reference_config_path}")
    else:
        config = VitsConfig(
            audio=audio_config,
            text_cleaner=text_cleaner,
            use_phonemes=COMMON_USE_PHONEMES,
            phonemizer=phonemizer,
            phoneme_language=phoneme_language,
            characters=characters,
        )
        print("[config] Starting a new VITS model.")

    return _apply_finetune_profile(
        config=config,
        run_name=run_name,
        output_path=output_path,
        dataset_config=dataset_config,
        audio_config=audio_config,
        text_cleaner=text_cleaner,
        phonemizer=phonemizer,
        phoneme_language=phoneme_language,
        test_sentences=test_sentences or [],
        characters=characters,
        epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        num_eval_workers=num_eval_workers,
        lr_gen=lr_gen,
        lr_disc=lr_disc,
    )


def run_training(
    *,
    config: VitsConfig,
    dataset_config: BaseDatasetConfig,
    output_path: str,
    continue_path: Optional[str] = None,
    restore_path: Optional[str] = None,
) -> None:
    os.makedirs(output_path, exist_ok=True)
    if config.phoneme_cache_path:
        os.makedirs(config.phoneme_cache_path, exist_ok=True)

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    print(f"[data] train_samples={len(train_samples)}")
    print(f"[data] eval_samples={len(eval_samples)}")
    print(f"[train] lr_disc={config.lr_disc:.8f}, lr_gen={config.lr_gen:.8f}")
    print(f"[train] target_loss={config.target_loss}")

    model = Vits(config, ap, tokenizer, speaker_manager=None)

    trainer_args = TrainerArgs()
    if continue_path:
        trainer_args = TrainerArgs(continue_path=continue_path)
    elif restore_path:
        trainer_args = TrainerArgs(restore_path=restore_path)

    trainer = Trainer(
        trainer_args,
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()
