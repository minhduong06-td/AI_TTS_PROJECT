import logging
from collections.abc import Callable
from typing import Union
from TTS.tts.utils.text import cleaners
from TTS.tts.utils.text.characters import BaseCharacters, Graphemes, IPAPhonemes
from TTS.tts.utils.text.phonemizers import DEF_LANG_TO_PHONEMIZER, get_phonemizer_by_name
from TTS.utils.generic_utils import get_import_path, import_class

logger = logging.getLogger(__name__)


class TTSTokenizer:
    def __init__(
        self,
        use_phonemes: bool = False,
        text_cleaner: Callable[[str], str] | None = None,
        characters: BaseCharacters | None = None,
        phonemizer: Union["Phonemizer", dict] | None = None,
        add_blank: bool = False,
        use_eos_bos: bool = False,
    ):
        self.text_cleaner = text_cleaner
        self.use_phonemes = use_phonemes
        self.add_blank = add_blank
        self.use_eos_bos = use_eos_bos
        self.characters = characters
        self.not_found_characters = []
        self.phonemizer = phonemizer

    @property
    def characters(self):
        return self._characters

    @characters.setter
    def characters(self, new_characters):
        self._characters = new_characters
        self.pad_id = self.characters.char_to_id(self.characters.pad) if self.characters.pad else None
        self.blank_id = self.characters.char_to_id(self.characters.blank) if self.characters.blank else None

    def encode(self, text: str) -> list[int]:
        token_ids = []
        for char in text:
            try:
                idx = self.characters.char_to_id(char)
                token_ids.append(idx)
            except KeyError:
                # discard but store not found characters
                if char not in self.not_found_characters:
                    self.not_found_characters.append(char)
                    logger.warning(text)
                    logger.warning("Character %s not found in the vocabulary. Discarding it.", repr(char))
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        text = ""
        for token_id in token_ids:
            text += self.characters.id_to_char(token_id)
        return text

    def text_to_ids(self, text: str, language: str | None = None) -> list[int]: 
        logger.debug("Tokenizer input text: %s", text)
        if self.text_cleaner is not None:
            text = self.text_cleaner(text)
            logger.debug("Cleaned text: %s", text)
        if self.use_phonemes:
            text = self.phonemizer.phonemize(text, separator="", language=language)
            logger.debug("Phonemes: %s", text)
        text = self.encode(text)
        if self.add_blank:
            text = self.intersperse_blank_char(text)
        if self.use_eos_bos:
            text = self.pad_with_bos_eos(text)
        return text

    def ids_to_text(self, id_sequence: list[int]) -> str:
        return self.decode(id_sequence)

    def pad_with_bos_eos(self, char_sequence: list[str]):
        return [self.characters.bos_id] + list(char_sequence) + [self.characters.eos_id]

    def intersperse_blank_char(self, char_sequence: list[str]):
        result = [self.characters.blank_id] * (len(char_sequence) * 2 + 1)
        result[1::2] = char_sequence
        return result

    def print_logs(self, level: int = 0):
        indent = "\t" * level
        logger.info("%s| add_blank: %s", indent, self.add_blank)
        logger.info("%s| use_eos_bos: %s", indent, self.use_eos_bos)
        logger.info("%s| use_phonemes: %s", indent, self.use_phonemes)
        if self.use_phonemes:
            logger.info("%s| phonemizer:", indent)
            self.phonemizer.print_logs(level + 1)
        if len(self.not_found_characters) > 0:
            logger.info("%s| %d characters not found:", indent, len(self.not_found_characters))
            for char in self.not_found_characters:
                logger.info("%s| %s", indent, char)

    @staticmethod
    def init_from_config(config: "Coqpit", characters: BaseCharacters | None = None):
        text_cleaner = None
        if isinstance(config.text_cleaner, str | list):
            text_cleaner = getattr(cleaners, config.text_cleaner)

        if characters is None:
            if config.characters and config.characters.characters_class:
                CharactersClass = import_class(config.characters.characters_class)
                if not issubclass(CharactersClass, BaseCharacters):
                    msg = f"{config.characters.characters_class} is not a subclass of BaseCharacters."
                    raise TypeError(msg)
                characters, new_config = CharactersClass.init_from_config(config)
            else:
                if config.use_phonemes:
                    characters, new_config = IPAPhonemes().init_from_config(config)
                else:
                    characters, new_config = Graphemes().init_from_config(config)

        else:
            characters, new_config = characters.init_from_config(config)

        new_config.characters.characters_class = get_import_path(characters)
        phonemizer = None
        if config.use_phonemes:
            phonemizer_kwargs = {"language": config.phoneme_language}
            if "phonemizer" in config and config.phonemizer:
                phonemizer = get_phonemizer_by_name(config.phonemizer, **phonemizer_kwargs)
            else:
                try:
                    phonemizer = get_phonemizer_by_name(
                        DEF_LANG_TO_PHONEMIZER[config.phoneme_language], **phonemizer_kwargs
                    )
                    new_config.phonemizer = phonemizer.name()
                except KeyError as e:
                    raise ValueError(
                        f"""No phonemizer found for language {config.phoneme_language}.
                        You may need to install a third party library for this language."""
                    ) from e

        return (
            TTSTokenizer(
                config.use_phonemes, text_cleaner, characters, phonemizer, config.add_blank, config.enable_eos_bos_chars
            ),
            new_config,
        )
