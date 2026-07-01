import abc
import logging
from typing import Any

from TTS.tts.utils.text.punctuation import _PUNC_IDX, Punctuation

logger = logging.getLogger(__name__)


class BasePhonemizer(abc.ABC):
    def __init__(
        self, language: str, punctuations: str = Punctuation.default_puncs(), *, keep_puncs: bool = False
    ) -> None:
        if not self.is_available():
            raise RuntimeError(f"{self.name()} not installed on your system")
        self._language = self._init_language(language)
        self._keep_puncs = keep_puncs
        self._punctuator = Punctuation(punctuations)

    def _init_language(self, language: str) -> str:
        if not self.is_supported_language(language):
            raise RuntimeError(f'language "{language}" is not supported by the {self.name()} backend')
        return language

    @property
    def language(self) -> str:
        return self._language

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        ...

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        ...

    @abc.abstractmethod
    def version(self) -> str:
        ...

    @staticmethod
    @abc.abstractmethod
    def supported_languages() -> list[str] | dict[str, Any]:
        ...

    def is_supported_language(self, language: str) -> bool:
        return language in self.supported_languages()

    @abc.abstractmethod
    def _phonemize(self, text: str, separator: str) -> str:
        """The main phonemization method"""

    def _phonemize_preprocess(self, text: str) -> tuple[list[str], list[_PUNC_IDX]]:
        text = text.strip()
        if self._keep_puncs:
            return self._punctuator.strip_to_restore(text)
        return [self._punctuator.strip(text)], []

    def _phonemize_postprocess(self, phonemized: list[str], punctuations: list[_PUNC_IDX]) -> str:
        if self._keep_puncs:
            return self._punctuator.restore(phonemized, punctuations)[0]
        return phonemized[0]

    def phonemize(self, text: str, separator: str = "|", language: str | None = None) -> str:
        preprocessed, punctuations = self._phonemize_preprocess(text)
        phonemized = []
        for t in preprocessed:
            p = self._phonemize(t, separator)
            phonemized.append(p)
        return self._phonemize_postprocess(phonemized, punctuations)

    def print_logs(self, level: int = 0) -> None:
        indent = "\t" * level
        logger.info("%s| phoneme language: %s", indent, self.language)
        logger.info("%s| phoneme backend: %s", indent, self.name())
