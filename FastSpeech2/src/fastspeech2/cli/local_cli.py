#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_HEADER_IDS = {"id", "filename", "file", "wav", "utterance_id"}


def fail(message):
    raise SystemExit(f"ERROR: {message}")


def run(command):
    printable = " ".join(str(item) for item in command)
    print(f"+ {printable}")
    subprocess.run([str(item) for item in command], cwd=ROOT, check=True)


def remove_path(path):
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def project_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def relative_to_root(path):
    try:
        return "./" + path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def dataset_config_dir(name):
    return ROOT / "configs" / "local" / name


def load_yaml(path):
    if not path.is_file():
        fail(f"Missing YAML file: {path}")
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        fail(f"Invalid YAML mapping: {path}")
    return data


def save_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)


def read_config(name, filename):
    path = dataset_config_dir(name) / filename
    return path, load_yaml(path)


def validate_name(name):
    if not NAME_RE.fullmatch(name):
        fail("--name may contain only letters, numbers, dot, underscore and hyphen")
    if name.endswith((".", "-")):
        fail("--name must not end with a dot or hyphen")


def parse_metadata(path):
    rows = []
    seen = set()
    first_record = True

    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            if line.count("|") not in (1, 2):
                fail(
                    f"metadata line {line_number}: expected exactly 2 or 3 pipe-separated fields"
                )

            fields = line.rstrip("\r\n").split("|")
            if len(fields) == 2:
                utterance_id, raw_text = fields
                normalized_text = raw_text
            else:
                utterance_id, raw_text, normalized_text = fields
                normalized_text = normalized_text or raw_text

            utterance_id = utterance_id.strip()
            if utterance_id.lower().endswith(".wav"):
                utterance_id = utterance_id[:-4]
            if first_record and utterance_id.casefold() in _HEADER_IDS:
                first_record = False
                continue
            first_record = False

            if not utterance_id or Path(utterance_id).name != utterance_id:
                fail(f"metadata line {line_number}: invalid id {utterance_id!r}")
            if any(ord(char) < 32 for char in utterance_id):
                fail(f"metadata line {line_number}: id contains a control character")

            key = utterance_id.casefold()
            if key in seen:
                fail(
                    f"metadata line {line_number}: duplicate id (case-insensitive) "
                    f"{utterance_id!r}"
                )
            if not raw_text.strip() and not normalized_text.strip():
                fail(f"metadata line {line_number}: empty text")

            seen.add(key)
            rows.append((utterance_id, raw_text.strip(), normalized_text.strip()))

    if not rows:
        fail("metadata contains no usable rows")
    return rows


def transfer_file(source, target, mode):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite staged WAV: {target}")

    if mode == "copy":
        shutil.copy2(source, target)
        return "copied"
    if mode == "hardlink":
        os.link(source, target)
        return "hardlinked"
    if mode == "symlink":
        target.symlink_to(source.resolve())
        return "symlinked"

    try:
        os.link(source, target)
        return "hardlinked"
    except OSError:
        pass
    if os.name != "nt":
        try:
            target.symlink_to(source.resolve())
            return "symlinked"
        except OSError:
            pass
    shutil.copy2(source, target)
    return "copied"


def artifact_paths(name):
    return {
        "config": dataset_config_dir(name),
        "corpus": ROOT / "workspace" / "corpus" / name,
        "raw": ROOT / "workspace" / "raw" / name,
        "preprocessed": ROOT / "workspace" / "preprocessed" / name,
        "checkpoint": ROOT / "outputs" / "checkpoints" / name,
        "log": ROOT / "outputs" / "logs" / name,
        "result": ROOT / "outputs" / "results" / name,
    }


def nonempty_checkpoint_dir(path):
    return path.exists() and any(
        item.is_file() or item.is_symlink() for item in path.rglob("*")
    )


def command_init(args):
    validate_name(args.name)
    if args.val_size is not None and args.val_size <= 0:
        fail("--val-size must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        fail("--batch-size must be positive")

    metadata = Path(args.metadata).expanduser().resolve()
    wavs = Path(args.wavs).expanduser().resolve()
    if not metadata.is_file():
        fail(f"metadata file not found: {metadata}")
    if not wavs.is_dir():
        fail(f"WAV directory not found: {wavs}")

    rows = parse_metadata(metadata)
    missing = [
        utterance_id
        for utterance_id, _, _ in rows
        if not (wavs / f"{utterance_id}.wav").is_file()
    ]
    if missing:
        preview = ", ".join(missing[:10])
        fail(f"{len(missing)} WAV files are missing; first ids: {preview}")

    auto_val = min(512, max(1, round(len(rows) * 0.02)))
    val_size = args.val_size if args.val_size is not None else auto_val
    if val_size > len(rows) - 5:
        fail(
            f"Validation size {val_size} leaves fewer than 5 training utterances "
            f"from {len(rows)} rows"
        )

    paths = artifact_paths(args.name)
    existing = [f"{label}: {path}" for label, path in paths.items() if path.exists()]
    if existing and not args.force:
        fail(
            "Dataset name already has artifacts. Use another --name or rerun with --force.\n"
            + "\n".join(existing)
        )
    if args.force and nonempty_checkpoint_dir(paths["checkpoint"]):
        fail(
            "--force refuses to replace a dataset that has checkpoints. "
            "Use another --name or move/delete the checkpoint directory manually."
        )

    base_name = "en_ljspeech" if args.language == "en" else "vi_ljspeech"
    base_dir = ROOT / "configs" / "templates" / base_name
    preprocess = load_yaml(base_dir / "preprocess.yaml")
    model = load_yaml(base_dir / "model.yaml")
    train = load_yaml(base_dir / "train.yaml")

    corpus = paths["corpus"]
    config_dir = paths["config"]
    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staged_corpus = corpus.parent / f".{args.name}.init-{token}"
    staged_config = config_dir.parent / f".{args.name}.init-{token}"
    remove_path(staged_corpus)
    remove_path(staged_config)

    transfer_counts = {
        "copied": 0,
        "hardlinked": 0,
        "symlinked": 0,
    }

    try:
        staged_wavs = staged_corpus / "wavs"
        staged_wavs.mkdir(parents=True)
        with (staged_corpus / "metadata.csv").open("w", encoding="utf-8") as stream:
            for utterance_id, raw_text, normalized_text in rows:
                stream.write(f"{utterance_id}|{raw_text}|{normalized_text}\n")

        for utterance_id, _, _ in rows:
            result = transfer_file(
                wavs / f"{utterance_id}.wav",
                staged_wavs / f"{utterance_id}.wav",
                args.link_mode,
            )
            transfer_counts[result] += 1

        raw_path = ROOT / "workspace" / "raw" / args.name
        preprocessed_path = ROOT / "workspace" / "preprocessed" / args.name
        preprocess["dataset"] = "CSVWav"
        preprocess["path"]["corpus_path"] = relative_to_root(corpus)
        preprocess["path"]["raw_path"] = relative_to_root(raw_path)
        preprocess["path"]["preprocessed_path"] = relative_to_root(
            preprocessed_path
        )
        preprocess["preprocessing"]["text"]["language"] = args.language
        preprocess["preprocessing"]["text"]["text_cleaners"] = [
            "english_cleaners" if args.language == "en" else "vietnamese_cleaners"
        ]
        preprocess["path"]["lexicon_path"] = (
            "assets/lexicons/en/librispeech-lexicon.txt"
            if args.language == "en"
            else "assets/lexicons/vi/vi-new-lexicon.txt"
        )
        preprocess["preprocessing"]["val_size"] = val_size

        model["lang"] = args.language
        model["multi_speaker"] = False
        model["vocoder"]["model"] = "HiFi-GAN"
        model["vocoder"]["speaker"] = (
            "LJSpeech" if args.language == "en" else "universal"
        )

        train["path"]["ckpt_path"] = f"./outputs/checkpoints/{args.name}"
        train["path"]["log_path"] = f"./outputs/logs/{args.name}"
        train["path"]["result_path"] = f"./outputs/results/{args.name}"
        train_size = len(rows) - val_size
        requested_batch = (
            args.batch_size
            if args.batch_size is not None
            else int(train["optimizer"]["batch_size"])
        )
        max_batch = max(1, (train_size - 1) // 4)
        train["optimizer"]["batch_size"] = min(requested_batch, max_batch)

        staged_config.mkdir(parents=True)
        save_yaml(staged_config / "preprocess.yaml", preprocess)
        save_yaml(staged_config / "model.yaml", model)
        save_yaml(staged_config / "train.yaml", train)

        if args.force:
            for path in paths.values():
                remove_path(path)

        corpus.parent.mkdir(parents=True, exist_ok=True)
        config_dir.parent.mkdir(parents=True, exist_ok=True)
        staged_corpus.replace(corpus)
        try:
            staged_config.replace(config_dir)
        except Exception:
            remove_path(corpus)
            raise
    except Exception:
        remove_path(staged_corpus)
        remove_path(staged_config)
        raise

    print(f"Dataset: {args.name}")
    print(f"Language: {args.language}")
    print(f"Utterances: {len(rows)}")
    print(f"Validation size: {val_size}")
    print(f"Batch size: {train['optimizer']['batch_size']}")
    print("WAV transfer: " + ", ".join(f"{key}={value}" for key, value in transfer_counts.items()))
    print(f"Config: {config_dir}")
    print("Next: python local_cli.py prepare --name " + args.name)


def command_prepare(args):
    preprocess_path, preprocess = read_config(args.name, "preprocess.yaml")
    raw_path = project_path(preprocess["path"]["raw_path"])
    speaker = Path(preprocess["path"]["preprocessed_path"]).name
    raw_speaker = raw_path / speaker
    if raw_speaker.exists() and not args.clean:
        fail(
            f"Prepared WAV/LAB directory already exists: {raw_speaker}. "
            "Use --clean to rebuild it deterministically."
        )
    if args.clean:
        remove_path(raw_speaker)
    run([sys.executable, "prepare_align.py", preprocess_path])


def command_align(args):
    _, preprocess = read_config(args.name, "preprocess.yaml")
    speaker = Path(preprocess["path"]["preprocessed_path"]).name
    raw_speaker = project_path(preprocess["path"]["raw_path"]) / speaker
    textgrid_dir = (
        project_path(preprocess["path"]["preprocessed_path"])
        / "TextGrid"
        / speaker
    )

    if not raw_speaker.is_dir():
        fail(f"Prepared WAV/LAB directory is missing: {raw_speaker}")
    wav_ids = {path.stem for path in raw_speaker.glob("*.wav")}
    lab_ids = {path.stem for path in raw_speaker.glob("*.lab")}
    if not wav_ids:
        fail(f"No WAV files found in {raw_speaker}")
    if wav_ids != lab_ids:
        missing_lab = sorted(wav_ids - lab_ids)
        missing_wav = sorted(lab_ids - wav_ids)
        fail(
            f"WAV/LAB mismatch: missing_lab={missing_lab[:10]}, "
            f"missing_wav={missing_wav[:10]}"
        )

    existing_textgrids = list(textgrid_dir.rglob("*.TextGrid")) if textgrid_dir.exists() else []
    if existing_textgrids and not args.clean_output:
        fail(
            f"Alignment output already contains {len(existing_textgrids)} TextGrid files. "
            "Use --clean-output to rebuild it."
        )
    if args.clean_output:
        remove_path(textgrid_dir)

    dictionary = args.dictionary
    acoustic_model = args.acoustic_model
    language = preprocess["preprocessing"]["text"]["language"]
    if language == "vi":
        dictionary = dictionary or str(ROOT / "assets" / "mfa" / "vi" / "vi-new-lexicon.dict")
        acoustic_model = acoustic_model or str(ROOT / "assets" / "mfa" / "vi" / "vi_new_mfa.zip")
        if not Path(dictionary).is_file():
            fail(f"Vietnamese MFA dictionary is missing: {dictionary}")
        if not Path(acoustic_model).is_file():
            fail(f"Vietnamese MFA acoustic model is missing: {acoustic_model}")
    if not dictionary or not acoustic_model:
        fail("English alignment requires --dictionary and --acoustic-model")

    textgrid_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.mfa_command,
        "align",
        raw_speaker,
        dictionary,
        acoustic_model,
        textgrid_dir,
    ]
    if args.clean:
        command.append("--clean")
    run(command)

    aligned_ids = {path.stem for path in textgrid_dir.rglob("*.TextGrid")}
    missing_ids = sorted(wav_ids - aligned_ids)
    print(
        f"Alignment: wav={len(wav_ids)}, textgrid={len(aligned_ids)}, "
        f"missing={len(missing_ids)}"
    )
    if not aligned_ids:
        fail("MFA completed but produced no TextGrid files")
    if missing_ids and not args.allow_partial:
        fail(
            f"MFA did not align {len(missing_ids)} utterances; first ids: "
            + ", ".join(missing_ids[:10])
            + ". Fix the data or use --allow-partial explicitly."
        )


def _feature_artifacts(preprocessed_path):
    return [
        preprocessed_path / "mel",
        preprocessed_path / "pitch",
        preprocessed_path / "energy",
        preprocessed_path / "duration",
        preprocessed_path / "train.txt",
        preprocessed_path / "val.txt",
        preprocessed_path / "stats.json",
        preprocessed_path / "speakers.json",
    ]


def count_nonempty_lines(path):
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def command_preprocess(args):
    preprocess_path, preprocess = read_config(args.name, "preprocess.yaml")
    train_path, train = read_config(args.name, "train.yaml")
    preprocessed_path = project_path(preprocess["path"]["preprocessed_path"])
    speaker = preprocessed_path.name
    textgrid_dir = preprocessed_path / "TextGrid" / speaker
    textgrids = list(textgrid_dir.rglob("*.TextGrid")) if textgrid_dir.exists() else []
    val_size = int(preprocess["preprocessing"]["val_size"])
    if len(textgrids) <= val_size + 4:
        fail(
            f"Only {len(textgrids)} aligned utterances are available, but val_size={val_size}. "
            "At least val_size + 5 aligned utterances are required."
        )

    artifacts = _feature_artifacts(preprocessed_path)
    existing = [path for path in artifacts if path.exists()]
    if existing and not args.clean:
        fail(
            "Preprocessed features already exist. Use --clean to rebuild them:\n"
            + "\n".join(str(path) for path in existing)
        )
    if args.clean:
        for path in artifacts:
            remove_path(path)

    estimated_train = len(textgrids) - val_size
    current_batch = int(train["optimizer"]["batch_size"])
    max_batch = max(1, (estimated_train - 1) // 4)
    if current_batch > max_batch:
        train["optimizer"]["batch_size"] = max_batch
        save_yaml(train_path, train)
        print(f"Adjusted batch size from {current_batch} to {max_batch}")

    run([sys.executable, "preprocess.py", preprocess_path])

    train_count = count_nonempty_lines(preprocessed_path / "train.txt")
    val_count = count_nonempty_lines(preprocessed_path / "val.txt")
    required_files = [
        preprocessed_path / "stats.json",
        preprocessed_path / "speakers.json",
    ]
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        fail("Preprocessing did not create required files: " + ", ".join(missing_files))
    batch_size = int(train["optimizer"]["batch_size"])
    if val_count == 0:
        fail("Preprocessing produced an empty validation set")
    if batch_size * 4 >= train_count:
        fail(
            f"Preprocessing produced train={train_count}, val={val_count}, "
            f"but batch_size={batch_size} violates batch_size*4 < train_size"
        )
    print(f"Preprocessed: train={train_count}, val={val_count}, batch_size={batch_size}")


def _validate_training_data(name):
    _, preprocess = read_config(name, "preprocess.yaml")
    _, train = read_config(name, "train.yaml")
    preprocessed_path = project_path(preprocess["path"]["preprocessed_path"])
    train_count = count_nonempty_lines(preprocessed_path / "train.txt")
    val_count = count_nonempty_lines(preprocessed_path / "val.txt")
    batch_size = int(train["optimizer"]["batch_size"])
    if train_count == 0 or val_count == 0:
        fail("Run the preprocess stage first; train.txt or val.txt is missing/empty")
    if batch_size * 4 >= train_count:
        fail(
            f"batch_size={batch_size} is too large for train_size={train_count}; "
            "require batch_size*4 < train_size"
        )
    return preprocess, train


def command_train(args):
    if args.restore_step < 0:
        fail("--restore-step must be non-negative")
    _, train = _validate_training_data(args.name)
    config_dir = dataset_config_dir(args.name)
    if args.restore_step:
        checkpoint = project_path(train["path"]["ckpt_path"]) / f"{args.restore_step}.pth.tar"
        if not checkpoint.is_file():
            fail(f"Checkpoint not found: {checkpoint}")

    command = [
        sys.executable,
        "train.py",
        "-p",
        config_dir / "preprocess.yaml",
        "-m",
        config_dir / "model.yaml",
        "-t",
        config_dir / "train.yaml",
        "--restore_step",
        str(args.restore_step),
    ]
    if args.no_tensorboard:
        command.append("--no_tensorboard")
    run(command)


def command_synthesize(args):
    if args.restore_step <= 0:
        fail("--restore-step must be positive for synthesis")
    if not args.text.strip():
        fail("--text must not be empty")
    if not NAME_RE.fullmatch(args.output_name):
        fail("--output-name may contain only letters, numbers, dot, underscore and hyphen")
    for label, value in (
        ("pitch", args.pitch_control),
        ("energy", args.energy_control),
        ("duration", args.duration_control),
    ):
        if value <= 0:
            fail(f"--{label}-control must be positive")

    _, preprocess = read_config(args.name, "preprocess.yaml")
    _, model = read_config(args.name, "model.yaml")
    _, train = read_config(args.name, "train.yaml")
    config_dir = dataset_config_dir(args.name)
    checkpoint = project_path(train["path"]["ckpt_path"]) / f"{args.restore_step}.pth.tar"
    if not checkpoint.is_file():
        fail(f"Checkpoint not found: {checkpoint}")

    preprocessed_path = project_path(preprocess["path"]["preprocessed_path"])
    for filename in ("stats.json", "speakers.json"):
        if not (preprocessed_path / filename).is_file():
            fail(f"Missing preprocessing artifact: {preprocessed_path / filename}")

    profile = model["vocoder"]["speaker"]
    vocoder_files = {
        "LJSpeech": ROOT / "assets" / "vocoders" / "hifigan" / "generator_LJSpeech.pth.tar",
        "universal": ROOT / "assets" / "vocoders" / "hifigan" / "generator_universal.pth.tar",
    }
    vocoder_file = vocoder_files.get(profile)
    if vocoder_file is None or not vocoder_file.is_file():
        fail(f"Missing/unsupported HiFi-GAN profile: {profile}")

    result_dir = project_path(train["path"]["result_path"])
    result_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "synthesize.py",
        "--restore_step",
        str(args.restore_step),
        "--mode",
        "single",
        "--text",
        args.text,
        "--output_name",
        args.output_name,
        "-p",
        config_dir / "preprocess.yaml",
        "-m",
        config_dir / "model.yaml",
        "-t",
        config_dir / "train.yaml",
        "--pitch_control",
        str(args.pitch_control),
        "--energy_control",
        str(args.energy_control),
        "--duration_control",
        str(args.duration_control),
    ]
    run(command)
    print(f"WAV: {result_dir / (args.output_name + '.wav')}")
    print(f"Plot: {result_dir / (args.output_name + '.png')}")


def command_status(args):
    preprocess_path, preprocess = read_config(args.name, "preprocess.yaml")
    _, train = read_config(args.name, "train.yaml")
    speaker = Path(preprocess["path"]["preprocessed_path"]).name
    preprocessed_path = project_path(preprocess["path"]["preprocessed_path"])
    paths = {
        "config": preprocess_path.parent,
        "corpus": project_path(preprocess["path"]["corpus_path"]),
        "raw speaker": project_path(preprocess["path"]["raw_path"]) / speaker,
        "TextGrid": preprocessed_path / "TextGrid" / speaker,
        "train.txt": preprocessed_path / "train.txt",
        "checkpoint dir": project_path(train["path"]["ckpt_path"]),
        "result dir": project_path(train["path"]["result_path"]),
    }
    for label, path in paths.items():
        print(f"{label:15} {'OK' if path.exists() else '--'}  {path}")
    print(f"train rows      {count_nonempty_lines(preprocessed_path / 'train.txt')}")
    print(f"val rows        {count_nonempty_lines(preprocessed_path / 'val.txt')}")
    textgrid_dir = preprocessed_path / "TextGrid" / speaker
    print(f"TextGrid files  {len(list(textgrid_dir.rglob('*.TextGrid'))) if textgrid_dir.exists() else 0}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="validate and stage metadata.csv + WAV files")
    init.add_argument("--name", required=True)
    init.add_argument("--language", choices=("en", "vi"), required=True)
    init.add_argument("--metadata", required=True)
    init.add_argument("--wavs", required=True)
    init.add_argument(
        "--link-mode",
        choices=("auto", "copy", "hardlink", "symlink"),
        default="auto",
    )
    init.add_argument("--val-size", type=int)
    init.add_argument("--batch-size", type=int)
    init.add_argument(
        "--force",
        action="store_true",
        help="replace existing non-checkpoint artifacts for this dataset name",
    )
    init.set_defaults(func=command_init)

    prepare = sub.add_parser("prepare", help="create normalized WAV/LAB pairs")
    prepare.add_argument("--name", required=True)
    prepare.add_argument("--clean", action="store_true")
    prepare.set_defaults(func=command_prepare)

    align = sub.add_parser("align", help="run Montreal Forced Aligner")
    align.add_argument("--name", required=True)
    align.add_argument("--dictionary")
    align.add_argument("--acoustic-model")
    align.add_argument("--mfa-command", default="mfa")
    align.add_argument("--clean", action="store_true", help="pass --clean to MFA")
    align.add_argument(
        "--clean-output",
        action="store_true",
        help="remove existing TextGrid output before alignment",
    )
    align.add_argument(
        "--allow-partial",
        action="store_true",
        help="accept fewer TextGrid files than input WAV files",
    )
    align.set_defaults(func=command_align)

    preprocess = sub.add_parser(
        "preprocess", help="build mel/pitch/energy/duration features"
    )
    preprocess.add_argument("--name", required=True)
    preprocess.add_argument("--clean", action="store_true")
    preprocess.set_defaults(func=command_preprocess)

    train = sub.add_parser("train", help="train or resume FastSpeech2")
    train.add_argument("--name", required=True)
    train.add_argument("--restore-step", type=int, default=0)
    train.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="disable TensorBoard previews and avoid loading the vocoder during train",
    )
    train.set_defaults(func=command_train)

    synth = sub.add_parser("synthesize", help="synthesize one sentence")
    synth.add_argument("--name", required=True)
    synth.add_argument("--restore-step", type=int, required=True)
    synth.add_argument("--text", required=True)
    synth.add_argument("--output-name", default="test")
    synth.add_argument("--pitch-control", type=float, default=1.0)
    synth.add_argument("--energy-control", type=float, default=1.0)
    synth.add_argument("--duration-control", type=float, default=1.0)
    synth.set_defaults(func=command_synthesize)

    status = sub.add_parser("status", help="show pipeline artifacts")
    status.add_argument("--name", required=True)
    status.set_defaults(func=command_status)
    return parser


def main():
    arguments = build_parser().parse_args()
    arguments.func(arguments)


if __name__ == "__main__":
    main()
