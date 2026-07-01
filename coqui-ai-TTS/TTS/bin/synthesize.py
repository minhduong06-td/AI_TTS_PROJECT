#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from argparse import RawTextHelpFormatter
from pathlib import Path

description = """
Local-only TTS CLI for custom checkpoints.

Examples:
  python3 -m TTS.bin.synthesize \\
    --text "Hello world" \\
    --model_path /path/to/best_model.pth \\
    --config_path /path/to/config.json \\
    --out_path output.wav

If --config_path is omitted, it will try:
  <dirname(model_path)>/config.json
"""


def main():
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument("--text", type=str, default=None, help="Text to synthesize.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to checkpoint file.")
    parser.add_argument("--config_path", type=str, default=None, help="Path to config.json.")
    parser.add_argument("--out_path", type=str, default="tts_output.wav", help="Output wav path.")

    parser.add_argument("--use_cuda", action="store_true", help="Run on CUDA.")
    parser.add_argument("--device", type=str, default="cpu", help="Device name if not using --use_cuda.")

    parser.add_argument(
        "--split_sentences",
        dest="split_sentences",
        action="store_true",
        default=True,
        help="Split input text into sentences before synthesis.",
    )
    parser.add_argument(
        "--no_split_sentences",
        dest="split_sentences",
        action="store_false",
        help="Disable sentence splitting.",
    )
    parser.add_argument("--length_scale", type=float, default=None, help="VITS speed control. Larger = slower.")
    parser.add_argument("--noise_scale", type=float, default=None, help="VITS acoustic randomness.")
    parser.add_argument("--noise_scale_dp", type=float, default=None, help="VITS duration predictor randomness.")

    parser.add_argument("--speaker_idx", type=str, default=None, help="Speaker id/name for multispeaker model.")
    parser.add_argument("--language_idx", type=str, default=None, help="Language id/name for multilingual model.")
    parser.add_argument("--speaker_wav", nargs="+", default=None, help="Reference wav(s) for speaker encoder models.")
    parser.add_argument("--list_speaker_idxs", action="store_true", help="List speaker ids in loaded model.")
    parser.add_argument("--list_language_idxs", action="store_true", help="List language ids in loaded model.")

    args = parser.parse_args()

    if not args.text and not args.list_speaker_idxs and not args.list_language_idxs:
        parser.error("--text is required unless you are listing speaker/language ids.")

    config_path = args.config_path
    if config_path is None:
        config_path = str(Path(args.model_path).resolve().parent / "config.json")

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    from TTS.utils.synthesizer import Synthesizer

    device = "cuda" if args.use_cuda else args.device

    synthesizer = Synthesizer(
        tts_checkpoint=args.model_path,
        tts_config_path=config_path,
        use_cuda=(device == "cuda"),
    )
    if args.length_scale is not None and hasattr(synthesizer.tts_model, "length_scale"):
        synthesizer.tts_model.length_scale = args.length_scale

    if args.noise_scale is not None:
        if hasattr(synthesizer.tts_model, "inference_noise_scale"):
            synthesizer.tts_model.inference_noise_scale = args.noise_scale
        if hasattr(synthesizer.tts_model, "noise_scale"):
            synthesizer.tts_model.noise_scale = args.noise_scale

    if args.noise_scale_dp is not None:
        if hasattr(synthesizer.tts_model, "inference_noise_scale_dp"):
            synthesizer.tts_model.inference_noise_scale_dp = args.noise_scale_dp
        if hasattr(synthesizer.tts_model, "noise_scale_dp"):
            synthesizer.tts_model.noise_scale_dp = args.noise_scale_dp

    if args.list_speaker_idxs:
        sm = getattr(synthesizer.tts_model, "speaker_manager", None)
        print(" > Available speaker ids:")
        print({} if sm is None or sm.name_to_id is None else sm.name_to_id)
        return

    if args.list_language_idxs:
        lm = getattr(synthesizer.tts_model, "language_manager", None)
        print(" > Available language ids:")
        print({} if lm is None or lm.name_to_id is None else lm.name_to_id)
        return

    print(f" > Text: {args.text}")

    wav = synthesizer.tts(
        text=args.text,
        speaker_name=args.speaker_idx,
        language_name=args.language_idx,
        speaker_wav=args.speaker_wav,
        split_sentences=args.split_sentences,
    )

    print(f" > Saving output to {args.out_path}")
    synthesizer.save_wav(wav, args.out_path)


if __name__ == "__main__":
    main()