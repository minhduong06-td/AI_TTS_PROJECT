from typing import Any

from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.phonemizers.espeak_wrapper import ESpeak

PHONEMIZERS = {ESpeak.name(): ESpeak}

ESPEAK_LANGS = list(ESpeak.supported_languages().keys())

DEF_LANG_TO_PHONEMIZER = {lang: ESpeak.name() for lang in ESPEAK_LANGS}

if "en-us" in DEF_LANG_TO_PHONEMIZER:
    DEF_LANG_TO_PHONEMIZER["en"] = DEF_LANG_TO_PHONEMIZER["en-us"]

DEF_LANG_TO_PHONEMIZER["vi"] = ESpeak.name()


def get_phonemizer_by_name(name: str, **kwargs: Any) -> BasePhonemizer:
    if name == "espeak":
        return ESpeak(**kwargs)
    raise ValueError(f"Phonemizer {name} not found")


if __name__ == "__main__":
    print(DEF_LANG_TO_PHONEMIZER)