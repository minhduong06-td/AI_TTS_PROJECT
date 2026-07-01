import re
from unicodedata import normalize

from .english.abbreviations import abbreviations_en
from .english.number_norm import normalize_numbers as en_normalize_numbers
from .english.time_norm import expand_time_english
from .vietnamese.abbreviations import abbreviations_vi
from .vietnamese.number_norm import normalize_numbers as vi_normalize_numbers
from .vietnamese.time_norm import expand_time_vietnamese
_whitespace_re = re.compile(r"\s+")
_uroman = None


def expand_abbreviations(text: str, lang: str = "en") -> str:
    if lang == "en":
        abbreviations = abbreviations_en
    elif lang == "vi":
        abbreviations = abbreviations_vi
    else:
        msg = f"Language {lang} not supported in expand_abbreviations"
        raise ValueError(msg)

    for regex, replacement in abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text: str) -> str:
    return text.lower()


def collapse_whitespace(text: str) -> str:
    return re.sub(_whitespace_re, " ", text).strip()


def romanize(text: str, language: str | None = None) -> str:
    global _uroman

    if _uroman is None:
        try:
            import uroman
        except ImportError as e:
            msg = "Package not installed: uroman."
            raise ImportError(msg) from e
        _uroman = uroman.Uroman()

    return _uroman.romanize_string(text, lcode=language)


def remove_aux_symbols(text: str) -> str:
    return re.sub(r"[\<\>\(\)\[\]\"]+", "", text)


def replace_symbols(text: str, lang: str | None = "en") -> str:
    text = text.replace(";", ",")
    text = text.replace("-", " ") if lang != "ca" else text.replace("-", "")
    text = text.replace(":", ",")

    if lang == "en":
        text = text.replace("&", " and ")

    return text


def basic_cleaners(text: str) -> str:
    text = normalize_unicode(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def uroman_cleaners(text: str) -> str:
    text = normalize_unicode(text)
    text = romanize(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def vietnamese_cleaners(text: str) -> str:
    text = normalize_unicode(text)
    text = lowercase(text)
    text = expand_time_vietnamese(text)
    text = vi_normalize_numbers(text)
    text = expand_abbreviations(text, lang="vi")
    text = replace_symbols(text, lang=None)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text: str) -> str:
    text = normalize_unicode(text)
    text = lowercase(text)
    text = expand_time_english(text)
    text = en_normalize_numbers(text)
    text = expand_abbreviations(text)
    text = replace_symbols(text)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text


def normalize_unicode(text: str) -> str:
    return normalize("NFC", text)