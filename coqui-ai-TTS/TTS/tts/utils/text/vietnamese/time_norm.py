import re

_time_colon_re = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_time_h_re = re.compile(r"\b([01]?\d|2[0-3])h(?:([0-5]\d))?\b", re.IGNORECASE)

_ones = [
    "không", "một", "hai", "ba", "bốn", "năm",
    "sáu", "bảy", "tám", "chín",
]

_teens = [
    "mười", "mười một", "mười hai", "mười ba", "mười bốn", "mười lăm",
    "mười sáu", "mười bảy", "mười tám", "mười chín",
]

def _num_to_words(n: int) -> str:
    assert 0 <= n <= 59
    if n < 10:
        return _ones[n]
    if n < 20:
        return _teens[n - 10]
    chuc = n // 10
    don = n % 10
    prefix = _ones[chuc] + " mươi"
    if don == 0:
        return prefix
    if don == 1:
        return prefix + " mốt"
    if don == 5:
        return prefix + " lăm"
    return prefix + " " + _ones[don]

def _format_time(hour: int, minute: int | None) -> str:
    result = _num_to_words(hour) + " giờ"
    if minute is not None and minute > 0:
        result += " " + _num_to_words(minute) + " phút"
    return result

def _expand_time_colon(match: "re.Match") -> str:
    hour = int(match.group(1))
    minute = int(match.group(2))
    return _format_time(hour, minute)

def _expand_time_h(match: "re.Match") -> str:
    hour = int(match.group(1))
    minute = match.group(2)
    return _format_time(hour, None if minute is None else int(minute))

def expand_time_vietnamese(text: str) -> str:
    text = re.sub(_time_colon_re, _expand_time_colon, text)
    text = re.sub(_time_h_re, _expand_time_h, text)
    return text