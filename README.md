# AI TTS Project: Coqui VITS vs FastSpeech2

This repository contains a local Text-to-Speech (TTS) project comparing two model families:

- **Coqui VITS**: an end-to-end text-to-waveform model.
- **FastSpeech2**: a non-autoregressive acoustic model that generates mel-spectrograms, followed by an external vocoder to synthesize waveform audio.

The project supports two languages:

- English: `en_ljspeech`
- Vietnamese: `vi_ljspeech`

Large datasets, preprocessed features and checkpoints are not committed directly to Git. Instead, placeholder `.txt` files are provided with download links.

---

## 1. Repository Structure

```text
AI_TTS_PROJECT/
├── coqui-ai-TTS/                  # Coqui VITS source and local training recipes
│   └── recipes/local/
│       ├── train_vits_en.py        # VITS English training recipe
│       ├── train_vits_vi.py        # VITS Vietnamese training recipe
│       └── vits_common.py          # shared VITS config/training helpers
│
├── FastSpeech2/                   # FastSpeech2 source code
│   ├── local_cli.py                # recommended local EN/VI pipeline CLI
│   ├── prepare_align.py            # wrapper for alignment preparation
│   ├── preprocess.py               # wrapper for feature preprocessing
│   ├── train.py                    # wrapper for model training
│   ├── synthesize.py               # wrapper for inference
│   ├── configs/templates/
│   │   ├── en_ljspeech/
│   │   └── vi_ljspeech/
│   └── src/fastspeech2/            # actual Python package source
│
├── web/                            # local web demo
├── FastSpeech2_runs/checkpoints/   # checkpoint download-link placeholders
├── coqui_runs/                     # Coqui run download-link placeholders
└── data/                           # local datasets, ignored by Git
```

---

## 2. Model Overview

### 2.1 Coqui VITS

VITS is an end-to-end neural TTS model. Instead of separating the acoustic model and the vocoder, VITS learns to generate waveform audio directly from text or phoneme input.

The main components are:

- **Text Encoder**: encodes text/phoneme tokens into a prior latent representation.
- **Posterior Encoder**: used during training to encode the real audio spectrogram into latent variable `z`.
- **Residual Coupling Flow**: connects the posterior latent space and the text prior latent space.
- **Monotonic Alignment Search (MAS)**: learns alignment between text tokens and audio frames without external TextGrid files.
- **Stochastic Duration Predictor**: predicts token durations during inference.
- **HiFi-GAN Generator**: synthesizes waveform audio directly.
- **Discriminator**: used during adversarial training to improve waveform naturalness.

In short:

```text
text / phoneme -> latent representation -> waveform
```

VITS is compact at inference time because it does not require a separate vocoder step. However, training is harder to debug because alignment, latent modeling, waveform generation and adversarial learning happen inside the same model.

### 2.2 FastSpeech2

FastSpeech2 is a non-autoregressive acoustic model. It does not synthesize waveform directly. Instead, it predicts a mel-spectrogram, and an external vocoder converts that mel-spectrogram into waveform audio.

The main components are:

- **Encoder**: encodes the input phoneme/text sequence.
- **Variance Adaptor**: predicts and injects duration, pitch and energy.
- **Length Regulator**: expands token-level representations according to duration.
- **Decoder**: generates hidden acoustic representations.
- **Mel Linear + PostNet**: produces and refines mel-spectrograms.
- **External HiFi-GAN Vocoder**: converts mel-spectrograms into waveform.

In short:

```text
text / phoneme -> mel-spectrogram -> waveform
```

FastSpeech2 is easier to debug than VITS because each target has a clear loss: mel, postnet mel, duration, pitch and energy. The downside is that it strongly depends on preprocessing quality, especially forced alignment and TextGrid accuracy.

---

## 3. Environment Setup

Recommended environment:

- Python 3.10 or 3.11
- CUDA-enabled PyTorch if training on GPU
- Montreal Forced Aligner for FastSpeech2 alignment
- espeak / phonemizer for phoneme processing

Example setup:

```bash
cd /home/md_dz6/AI

# Create and activate environment according to your local setup.
# Example only:
python3 -m venv .venv
source .venv/bin/activate

# Install FastSpeech2 dependencies
cd FastSpeech2
pip install -r requirements.txt

# Install Coqui TTS package in editable mode if needed
cd ../coqui-ai-TTS
pip install -e .
```

Install PyTorch separately according to the CUDA version of the machine.

---

## 4. Data Layout

Both models use LJSpeech-style data:

```text
data/
├── en_ljspeech/
│   ├── metadata.csv or metadata_clean.csv
│   └── wavs/
│       ├── <utterance_id>.wav
│       └── ...
└── vi_ljspeech/
    ├── metadata.csv or metadata_clean.csv
    └── wavs/
        ├── <utterance_id>.wav
        └── ...
```

Accepted metadata format:

```text
id|text
```

or:

```text
id|raw_text|normalized_text
```

For Coqui VITS recipes in this project, the expected metadata file is:

```text
metadata_clean.csv
```

---

## 5. Training Coqui VITS

### 5.1 Train VITS English

```bash
cd /home/md_dz6/AI/coqui-ai-TTS
python recipes/local/train_vits_en.py
```

Optional training controls:

```bash
python recipes/local/train_vits_en.py \
  --epochs 1000 \
  --batch_size 16 \
  --eval_batch_size 8 \
  --num_workers 4 \
  --num_eval_workers 2 \
  --lr_gen 0.0002 \
  --lr_disc 0.0002
```

Resume the same run:

```bash
python recipes/local/train_vits_en.py \
  --continue_path /home/md_dz6/AI/coqui_runs/en/en_ljspeech_vits_real/<run_directory>
```

Fine-tune from a checkpoint:

```bash
python recipes/local/train_vits_en.py \
  --restore_path /home/md_dz6/AI/coqui_runs/en/en_ljspeech_vits_real/<run_directory>/best_model.pth \
  --restore_config /home/md_dz6/AI/coqui_runs/en/en_ljspeech_vits_real/<run_directory>/config.json
```

### 5.2 Train VITS Vietnamese

```bash
cd /home/md_dz6/AI/coqui-ai-TTS
python recipes/local/train_vits_vi.py
```

Resume the same run:

```bash
python recipes/local/train_vits_vi.py \
  --continue_path /home/md_dz6/AI/coqui_runs/vi/vi_ljspeech_vits_real/<run_directory>
```

Fine-tune from a checkpoint:

```bash
python recipes/local/train_vits_vi.py \
  --restore_path /home/md_dz6/AI/coqui_runs/vi/vi_ljspeech_vits_real/<run_directory>/best_model.pth \
  --restore_config /home/md_dz6/AI/coqui_runs/vi/vi_ljspeech_vits_real/<run_directory>/config.json
```

---

## 6. Generating Audio with Coqui VITS

### 6.1 VITS English inference

```bash
cd /home/md_dz6/AI/coqui-ai-TTS

python -m TTS.bin.synthesize \
  --text "The quick brown fox jumps over the lazy dog." \
  --model_path /home/md_dz6/AI/coqui_runs/en/en_ljspeech_vits_real/<run_directory>/best_model.pth \
  --config_path /home/md_dz6/AI/coqui_runs/en/en_ljspeech_vits_real/<run_directory>/config.json \
  --out_path /home/md_dz6/AI/web/output/vits/en_test.wav \
  --device cuda
```

Use CPU if CUDA is not available:

```bash
python -m TTS.bin.synthesize \
  --text "The quick brown fox jumps over the lazy dog." \
  --model_path /home/md_dz6/AI/coqui_runs/en/en_ljspeech_vits_real/<run_directory>/best_model.pth \
  --config_path /home/md_dz6/AI/coqui_runs/en/en_ljspeech_vits_real/<run_directory>/config.json \
  --out_path /home/md_dz6/AI/web/output/vits/en_test.wav \
  --device cpu
```

### 6.2 VITS Vietnamese inference

```bash
cd /home/md_dz6/AI/coqui-ai-TTS

python -m TTS.bin.synthesize \
  --text "Xin chào, tôi đang thử nghiệm mô hình đọc tiếng Việt." \
  --model_path /home/md_dz6/AI/coqui_runs/vi/vi_ljspeech_vits_real/<run_directory>/best_model.pth \
  --config_path /home/md_dz6/AI/coqui_runs/vi/vi_ljspeech_vits_real/<run_directory>/config.json \
  --out_path /home/md_dz6/AI/web/output/vits/vi_test.wav \
  --device cuda
```

Optional VITS inference controls:

```bash
python -m TTS.bin.synthesize \
  --text "Xin chào, đây là câu kiểm tra." \
  --model_path <checkpoint.pth> \
  --config_path <config.json> \
  --out_path output.wav \
  --device cuda \
  --length_scale 1.0 \
  --noise_scale 0.35 \
  --noise_scale_dp 0.6
```

---

## 7. Training FastSpeech2

FastSpeech2 training is target-based. Before training, the pipeline must create:

- TextGrid alignment
- duration targets
- mel-spectrograms
- pitch targets
- energy targets
- statistics for normalization

The recommended interface is `local_cli.py`.

### 7.1 FastSpeech2 English pipeline

Initialize dataset:

```bash
cd /home/md_dz6/AI/FastSpeech2

python local_cli.py init \
  --name en_ljspeech \
  --language en \
  --metadata /home/md_dz6/AI/data/en_ljspeech/metadata.csv \
  --wavs /home/md_dz6/AI/data/en_ljspeech/wavs \
  --force
```

Prepare WAV/LAB files:

```bash
python local_cli.py prepare --name en_ljspeech --clean
```

Run forced alignment. Replace dictionary and acoustic model paths with the local MFA resources:

```bash
python local_cli.py align \
  --name en_ljspeech \
  --dictionary <path_to_english_dictionary> \
  --acoustic-model <path_to_english_acoustic_model> \
  --clean \
  --clean-output
```

Preprocess targets:

```bash
python local_cli.py preprocess --name en_ljspeech --clean
```

Train:

```bash
python local_cli.py train --name en_ljspeech
```

Resume training:

```bash
python local_cli.py train --name en_ljspeech --restore-step 85000
```

Train without TensorBoard/vocoder preview:

```bash
python local_cli.py train --name en_ljspeech --no-tensorboard
```

### 7.2 FastSpeech2 Vietnamese pipeline

Initialize dataset:

```bash
cd /home/md_dz6/AI/FastSpeech2

python local_cli.py init \
  --name vi_ljspeech \
  --language vi \
  --metadata /home/md_dz6/AI/data/vi_ljspeech/metadata.csv \
  --wavs /home/md_dz6/AI/data/vi_ljspeech/wavs \
  --force
```

Prepare WAV/LAB files:

```bash
python local_cli.py prepare --name vi_ljspeech --clean
```

Run forced alignment. The Vietnamese pipeline can use the bundled MFA dictionary and acoustic model when available:

```bash
python local_cli.py align --name vi_ljspeech --clean --clean-output
```

Preprocess targets:

```bash
python local_cli.py preprocess --name vi_ljspeech --clean
```

Train:

```bash
python local_cli.py train --name vi_ljspeech
```

Resume training:

```bash
python local_cli.py train --name vi_ljspeech --restore-step 220000
```

---

## 8. Generating Audio with FastSpeech2

### 8.1 FastSpeech2 English inference

```bash
cd /home/md_dz6/AI/FastSpeech2

python local_cli.py synthesize \
  --name en_ljspeech \
  --restore-step 85000 \
  --text "The quick brown fox jumps over the lazy dog." \
  --output-name en_test
```

### 8.2 FastSpeech2 Vietnamese inference

```bash
cd /home/md_dz6/AI/FastSpeech2

python local_cli.py synthesize \
  --name vi_ljspeech \
  --restore-step 220000 \
  --text "Xin chào, tôi đang thử nghiệm mô hình FastSpeech2 tiếng Việt." \
  --output-name vi_test
```

Inference controls:

```bash
python local_cli.py synthesize \
  --name vi_ljspeech \
  --restore-step 220000 \
  --text "Câu kiểm tra tốc độ, cao độ và năng lượng." \
  --output-name vi_control_test \
  --pitch-control 1.0 \
  --energy-control 1.0 \
  --duration-control 1.0
```

---

## 9. Low-Level FastSpeech2 Commands

The local CLI wraps the lower-level scripts. If needed, the pipeline can also be executed directly with config files.

English:

```bash
cd /home/md_dz6/AI/FastSpeech2

python prepare_align.py configs/templates/en_ljspeech/preprocess.yaml
python preprocess.py configs/templates/en_ljspeech/preprocess.yaml

python train.py \
  -p configs/templates/en_ljspeech/preprocess.yaml \
  -m configs/templates/en_ljspeech/model.yaml \
  -t configs/templates/en_ljspeech/train.yaml

python synthesize.py \
  --restore_step 85000 \
  --mode single \
  --text "The quick brown fox jumps over the lazy dog." \
  --output_name en_test \
  -p configs/templates/en_ljspeech/preprocess.yaml \
  -m configs/templates/en_ljspeech/model.yaml \
  -t configs/templates/en_ljspeech/train.yaml
```

Vietnamese:

```bash
cd /home/md_dz6/AI/FastSpeech2

python prepare_align.py configs/templates/vi_ljspeech/preprocess.yaml
python preprocess.py configs/templates/vi_ljspeech/preprocess.yaml

python train.py \
  -p configs/templates/vi_ljspeech/preprocess.yaml \
  -m configs/templates/vi_ljspeech/model.yaml \
  -t configs/templates/vi_ljspeech/train.yaml

python synthesize.py \
  --restore_step 220000 \
  --mode single \
  --text "Xin chào, đây là câu kiểm tra tiếng Việt." \
  --output_name vi_test \
  -p configs/templates/vi_ljspeech/preprocess.yaml \
  -m configs/templates/vi_ljspeech/model.yaml \
  -t configs/templates/vi_ljspeech/train.yaml
```

---

## 10. Local Web Demo

The `web/` folder contains a local demo interface for running inference from the browser.

Typical usage:

```bash
cd /home/md_dz6/AI/web
pip install -r requirements.txt
python backend.py
```

Then open the frontend in a browser according to the local web configuration.

Generated audio is stored under:

```text
web/output/
```

---

## 11. Large Artifacts

Large artifacts are intentionally not committed directly:

- raw datasets
- preprocessed FastSpeech2 features
- checkpoints
- generated waveform files
- training logs

Instead, this repository contains small `.txt` files with Google Drive download links under:

```text
FastSpeech2/workspace/preprocessed/
FastSpeech2_runs/checkpoints/
coqui_runs/
```

---

## 12. Quick Command Summary

Train VITS English:

```bash
cd /home/md_dz6/AI/coqui-ai-TTS
python recipes/local/train_vits_en.py
```

Train VITS Vietnamese:

```bash
cd /home/md_dz6/AI/coqui-ai-TTS
python recipes/local/train_vits_vi.py
```

Generate VITS audio:

```bash
python -m TTS.bin.synthesize \
  --text "Hello from VITS." \
  --model_path <best_model.pth> \
  --config_path <config.json> \
  --out_path output.wav \
  --device cuda
```

Train FastSpeech2 English:

```bash
cd /home/md_dz6/AI/FastSpeech2
python local_cli.py train --name en_ljspeech
```

Train FastSpeech2 Vietnamese:

```bash
cd /home/md_dz6/AI/FastSpeech2
python local_cli.py train --name vi_ljspeech
```

Generate FastSpeech2 audio:

```bash
python local_cli.py synthesize \
  --name en_ljspeech \
  --restore-step 85000 \
  --text "Hello from FastSpeech2." \
  --output-name en_test
```

---

## 13. Notes

- VITS training should be evaluated with both loss curves and generated audio.
- FastSpeech2 loss curves are easier to interpret because every target is explicit.
- Vietnamese TTS must be checked by listening, especially for tone marks, final consonants and sentence rhythm.
- A low training loss does not always guarantee good inference quality.
