# FastSpeech2 local EN/VI CLI

This build keeps the FastSpeech2 model/loss logic and standardizes both languages on one dataset contract:

```text
metadata.csv
wavs/<utterance_id>.wav
```

Metadata is UTF-8 without a required header. Accepted rows are `id|text` or `id|raw_text|normalized_text`.

Recommended environment: Python 3.10 or 3.11. Install PyTorch separately for the CUDA version on the machine, then install `requirements-local.txt`. Montreal Forced Aligner is installed separately.

## Pipeline

```bash
python local_cli.py init --name my_en --language en --metadata /data/en/metadata.csv --wavs /data/en/wavs
python local_cli.py prepare --name my_en
python local_cli.py align --name my_en --dictionary <dict> --acoustic-model <model> --clean
python local_cli.py preprocess --name my_en
python local_cli.py train --name my_en
python local_cli.py synthesize --name my_en --restore-step 100000 --text "Local CLI test."
```

Vietnamese uses the bundled MFA dictionary and acoustic model:

```bash
python local_cli.py init --name my_vi --language vi --metadata /data/vi/metadata.csv --wavs /data/vi/wavs
python local_cli.py prepare --name my_vi
python local_cli.py align --name my_vi --clean
python local_cli.py preprocess --name my_vi
python local_cli.py train --name my_vi
```

Each destructive rebuild is explicit:

- `init --force` replaces an initialized dataset only when no checkpoints exist.
- `prepare --clean` rebuilds WAV/LAB output.
- `align --clean-output` removes old TextGrid output.
- `preprocess --clean` removes old mel/pitch/energy/duration features while preserving TextGrid files.

TensorBoard is enabled by default, matching the original training workflow. Add `--no-tensorboard` to `train` only when you need to save VRAM/RAM and do not need preview figures/audio.

Removing Pinyin changes the English symbol vocabulary. Existing English checkpoints made before this cleanup are incompatible unless they already used the same EN-only vocabulary.
