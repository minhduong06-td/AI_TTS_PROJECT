import logging
from dataclasses import replace
from TTS.tts.configs.shared_configs import BaseTTSConfig, CharactersConfig
logger = logging.getLogger(__name__)


def parse_symbols() -> dict[str, str]:
    return {
        "pad": _pad,
        "eos": _eos,
        "bos": _bos,
        "characters": _characters,
        "punctuations": _punctuations,
        "phonemes": _phonemes,
    }

_pad = "<PAD>"
_eos = "<EOS>"
_bos = "<BOS>"
_blank = "<BLNK>"  
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_punctuations = "!'(),-.:;? "
_vowels = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ"
_non_pulmonic_consonants = "ʘɓǀɗǃʄǂɠǁʛ"
_pulmonic_consonants = "pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ"
_suprasegmentals = "ˈˌːˑ"
_other_symbols = "ʍwɥʜʢʡɕʑɺɧʲ"
_diacritics = "̃ɚ˞ɫ"
_phonemes = _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacritics


class BaseVocabulary:
    def __init__(
        self, vocab: list[str] | None, pad: str = None, blank: str = None, bos: str = None, eos: str = None
    ) -> None:
        self.vocab = vocab
        self.pad = pad
        self.blank = blank
        self.bos = bos
        self.eos = eos

    @property
    def pad_id(self) -> int:
        return self.char_to_id(self.pad) if self.pad else len(self.vocab)

    @property
    def blank_id(self) -> int:
        return self.char_to_id(self.blank) if self.blank else len(self.vocab)

    @property
    def bos_id(self) -> int:
        return self.char_to_id(self.bos) if self.bos else len(self.vocab)

    @property
    def eos_id(self) -> int:
        return self.char_to_id(self.eos) if self.eos else len(self.vocab)

    @property
    def vocab(self) -> list[str]:
        return self._vocab

    @vocab.setter
    def vocab(self, vocab: list[str] | None) -> None:
        self._vocab, self._char_to_id, self._id_to_char = [], {}, {}
        if vocab is not None:
            self._vocab = vocab
            self._char_to_id = {char: idx for idx, char in enumerate(self._vocab)}
            self._id_to_char = dict(enumerate(self._vocab))

    @staticmethod
    def init_from_config(config: BaseTTSConfig, **kwargs) -> tuple["BaseVocabulary", BaseTTSConfig]:
        if config.characters is not None and "vocab_dict" in config.characters and config.characters.vocab_dict:
            return (
                BaseVocabulary(
                    config.characters.vocab_dict,
                    config.characters.pad,
                    config.characters.blank,
                    config.characters.bos,
                    config.characters.eos,
                ),
                config,
            )
        return BaseVocabulary(**kwargs), config

    def to_config(self) -> "CharactersConfig":
        return CharactersConfig(
            vocab_dict=self._vocab,
            pad=self.pad,
            eos=self.eos,
            bos=self.bos,
            blank=self.blank,
            is_unique=False,
            is_sorted=False,
        )

    @property
    def num_chars(self) -> int:
        return len(self._vocab)

    def char_to_id(self, char: str) -> int:
        """Map a character to an token ID."""
        try:
            return self._char_to_id[char]
        except KeyError as e:
            raise KeyError(msg) from e

    def id_to_char(self, idx: int) -> str:
        return self._id_to_char[idx]


class BaseCharacters:
    def __init__(
        self,
        characters: str = None,
        punctuations: str = None,
        pad: str = None,
        eos: str = None,
        bos: str = None,
        blank: str = None,
        *,
        is_unique: bool = False,
        is_sorted: bool = True,
    ) -> None:
        self._characters = characters
        self._punctuations = punctuations
        self._pad = pad
        self._eos = eos
        self._bos = bos
        self._blank = blank
        self.is_unique = is_unique
        self.is_sorted = is_sorted
        self._create_vocab()

    @property
    def pad_id(self) -> int:
        return self.char_to_id(self.pad) if self.pad else len(self.vocab)

    @property
    def blank_id(self) -> int:
        return self.char_to_id(self.blank) if self.blank else len(self.vocab)

    @property
    def eos_id(self) -> int:
        return self.char_to_id(self.eos) if self.eos else len(self.vocab)

    @property
    def bos_id(self) -> int:
        return self.char_to_id(self.bos) if self.bos else len(self.vocab)

    @property
    def characters(self) -> str:
        return self._characters

    @characters.setter
    def characters(self, characters: str) -> None:
        self._characters = characters
        self._create_vocab()

    @property
    def punctuations(self) -> str:
        return self._punctuations

    @punctuations.setter
    def punctuations(self, punctuations: str) -> None:
        self._punctuations = punctuations
        self._create_vocab()

    @property
    def pad(self) -> str:
        return self._pad

    @pad.setter
    def pad(self, pad: str) -> None:
        self._pad = pad
        self._create_vocab()

    @property
    def eos(self) -> str | None:
        return self._eos

    @eos.setter
    def eos(self, eos: str | None) -> None:
        self._eos = eos
        self._create_vocab()

    @property
    def bos(self) -> str | None:
        return self._bos

    @bos.setter
    def bos(self, bos: str | None) -> None:
        self._bos = bos
        self._create_vocab()

    @property
    def blank(self) -> str | None:
        return self._blank

    @blank.setter
    def blank(self, blank: str | None) -> None:
        self._blank = blank
        self._create_vocab()

    @property
    def vocab(self) -> list[str]:
        return self._vocab

    @vocab.setter
    def vocab(self, vocab: list[str]) -> None:
        self._vocab = vocab
        self._char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self._id_to_char = dict(enumerate(self.vocab))

    @property
    def num_chars(self) -> int:
        return len(self._vocab)

    def _create_vocab(self) -> None:
        _vocab = self._characters
        if self.is_unique:
            _vocab = list(set(_vocab))
        if self.is_sorted:
            _vocab = sorted(_vocab)
        _vocab = list(_vocab)
        _vocab = [self._blank, *_vocab] if self._blank is not None and len(self._blank) > 0 else _vocab
        _vocab = [self._bos, *_vocab] if self._bos is not None and len(self._bos) > 0 else _vocab
        _vocab = [self._eos, *_vocab] if self._eos is not None and len(self._eos) > 0 else _vocab
        _vocab = [self._pad, *_vocab] if self._pad is not None and len(self._pad) > 0 else _vocab
        self.vocab = _vocab + list(self._punctuations)
        if self.is_unique:
            duplicates = {x for x in self.vocab if self.vocab.count(x) > 1}
            assert len(self.vocab) == len(self._char_to_id) == len(self._id_to_char), (
                f"There are duplicate characters in the character set. {duplicates}"
            )

    def char_to_id(self, char: str) -> int:
        try:
            return self._char_to_id[char]
        except KeyError as e:
            msg = f"{repr(char)} is not in the vocabulary."
            raise KeyError(msg) from e

    def id_to_char(self, idx: int) -> str:
        return self._id_to_char[idx]

    def print_log(self, level: int = 0) -> None:
        indent = "\t" * level
        logger.info("%s| Characters: %s", indent, self._characters)
        logger.info("%s| Punctuations: %s", indent, self._punctuations)
        logger.info("%s| Pad: %s", indent, self._pad)
        logger.info("%s| EOS: %s", indent, self._eos)
        logger.info("%s| BOS: %s", indent, self._bos)
        logger.info("%s| Blank: %s", indent, self._blank)
        logger.info("%s| Vocab: %s", indent, self.vocab)
        logger.info("%s| Num chars: %d", indent, self.num_chars)

    @staticmethod
    def init_from_config(config: BaseTTSConfig) -> tuple["BaseCharacters", BaseTTSConfig]:
        if config.characters is not None:
            return BaseCharacters(**config.characters), config
        characters = BaseCharacters()
        new_config = replace(config, characters=characters.to_config())
        return characters, new_config

    def to_config(self) -> "CharactersConfig":
        return CharactersConfig(
            characters=self._characters,
            punctuations=self._punctuations,
            pad=self._pad,
            eos=self._eos,
            bos=self._bos,
            blank=self._blank,
            is_unique=self.is_unique,
            is_sorted=self.is_sorted,
        )


class IPAPhonemes(BaseCharacters):
    def __init__(
        self,
        characters: str = _phonemes,
        punctuations: str = _punctuations,
        pad: str = _pad,
        eos: str = _eos,
        bos: str = _bos,
        blank: str = _blank,
        *,
        is_unique: bool = False,
        is_sorted: bool = True,
    ) -> None:
        super().__init__(characters, punctuations, pad, eos, bos, blank, is_unique=is_unique, is_sorted=is_sorted)

    @staticmethod
    def init_from_config(config: BaseTTSConfig) -> tuple["IPAPhonemes", BaseTTSConfig]:
        if "characters" in config and config.characters is not None:
            if "phonemes" in config.characters and config.characters.phonemes is not None:
                config.characters["characters"] = config.characters["phonemes"]
            return (
                IPAPhonemes(
                    characters=config.characters["characters"],
                    punctuations=config.characters["punctuations"],
                    pad=config.characters["pad"],
                    eos=config.characters["eos"],
                    bos=config.characters["bos"],
                    blank=config.characters["blank"],
                    is_unique=config.characters["is_unique"],
                    is_sorted=config.characters["is_sorted"],
                ),
                config,
            )
        # use character set from config
        if config.characters is not None:
            return IPAPhonemes(**config.characters), config
        # return default character set
        characters = IPAPhonemes()
        new_config = replace(config, characters=characters.to_config())
        return characters, new_config


class Graphemes(BaseCharacters):
    def __init__(
        self,
        characters: str = _characters,
        punctuations: str = _punctuations,
        pad: str = _pad,
        eos: str = _eos,
        bos: str = _bos,
        blank: str = _blank,
        *,
        is_unique: bool = False,
        is_sorted: bool = True,
    ) -> None:
        super().__init__(characters, punctuations, pad, eos, bos, blank, is_unique=is_unique, is_sorted=is_sorted)

    @staticmethod
    def init_from_config(config: BaseTTSConfig) -> tuple["Graphemes", BaseTTSConfig]:
        if config.characters is not None:
            if "phonemes" in config.characters:
                return (
                    Graphemes(
                        characters=config.characters["characters"],
                        punctuations=config.characters["punctuations"],
                        pad=config.characters["pad"],
                        eos=config.characters["eos"],
                        bos=config.characters["bos"],
                        blank=config.characters["blank"],
                        is_unique=config.characters["is_unique"],
                        is_sorted=config.characters["is_sorted"],
                    ),
                    config,
                )
            return Graphemes(**config.characters), config
        characters = Graphemes()
        new_config = replace(config, characters=characters.to_config())
        return characters, new_config


if __name__ == "__main__":
    gr = Graphemes()
    ph = IPAPhonemes()
    gr.print_log()
    ph.print_log()
