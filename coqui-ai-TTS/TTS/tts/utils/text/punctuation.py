import re
from enum import Enum
from typing import NamedTuple
import six
_DEF_PUNCS = ';:,.!?¡¿—…"«»“”'

class PuncPosition(Enum):
    BEGIN = 0
    END = 1
    MIDDLE = 2


class _PUNC_IDX(NamedTuple):
    punc: str
    position: PuncPosition


class Punctuation:
    def __init__(self, puncs: str = _DEF_PUNCS) -> None:
        self.puncs = puncs

    @staticmethod
    def default_puncs() -> str:
        return _DEF_PUNCS

    @property
    def puncs(self) -> str:
        return self._puncs

    @puncs.setter
    def puncs(self, value: str) -> None:
        if not isinstance(value, six.string_types):
            raise ValueError("Punctuations must be of type str.")
        self._puncs = "".join(list(dict.fromkeys(list(value)))) 
        self.puncs_regular_exp = re.compile(rf"(\s*[{re.escape(self._puncs)}]+\s*)+")

    def strip(self, text: str) -> str:
        return re.sub(self.puncs_regular_exp, " ", text).rstrip().lstrip()

    def strip_to_restore(self, text: str) -> tuple[list[str], list[_PUNC_IDX]]:
        stripped, puncs = self._strip_to_restore(text)
        return stripped, puncs

    def _strip_to_restore(self, text: str) -> tuple[list[str], list[_PUNC_IDX]]:
        matches = list(re.finditer(self.puncs_regular_exp, text))
        if not matches:
            return [text], []
        if len(matches) == 1 and matches[0].group() == text:
            return [], [_PUNC_IDX(text, PuncPosition.BEGIN)]
        puncs = []
        for match in matches:
            position = PuncPosition.MIDDLE
            if match == matches[0] and text.startswith(match.group()):
                position = PuncPosition.BEGIN
            elif match == matches[-1] and text.endswith(match.group()):
                position = PuncPosition.END
            puncs.append(_PUNC_IDX(match.group(), position))
        splitted_text = []
        for idx, punc in enumerate(puncs):
            split = text.split(punc.punc)
            prefix, suffix = split[0], punc.punc.join(split[1:])
            text = suffix
            if prefix == "":
                continue
            splitted_text.append(prefix)
            if idx == len(puncs) - 1 and len(suffix) > 0:
                splitted_text.append(suffix)
        return splitted_text, puncs

    @classmethod
    def restore(cls, text: list[str], puncs: list[_PUNC_IDX]) -> list[str]:
        return cls._restore(text, puncs)

    @classmethod
    def _restore(cls, text: list[str], puncs: list[_PUNC_IDX]) -> list[str]:
        if not puncs:
            return text
        if not text:
            return ["".join(m.punc for m in puncs)]

        current = puncs[0]

        if current.position == PuncPosition.BEGIN:
            return cls._restore([current.punc + text[0]] + text[1:], puncs[1:])

        if current.position == PuncPosition.END:
            return [text[0] + current.punc] + cls._restore(text[1:], puncs[1:])
        if len(text) == 1:  
            return cls._restore([text[0] + current.punc], puncs[1:])

        return cls._restore([text[0] + current.punc + text[1]] + text[2:], puncs[1:])


