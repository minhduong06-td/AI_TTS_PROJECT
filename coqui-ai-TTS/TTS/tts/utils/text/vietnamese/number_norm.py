import re
_date_re = re.compile(r"\b([0-3]?\d)[/\-]([01]?\d)[/\-](\d{4})\b")
_year_range_re = re.compile(r"\b(\d{4})\s*[-–]\s*(\d{4})\b")
_phone_re = re.compile(r"\b(?:\+?84|0)(?:[\s.\-]?\d){8,10}\b")
_percent_re = re.compile(r"([0-9]+(?:[\.,][0-9]+)?)\s*%")
_currency_re = re.compile(r"\b([0-9][0-9.,]*)\s*(đ|vnđ|vnd|usd|eur|gbp|jpy)\b", re.IGNORECASE)
_fraction_re = re.compile(r"\b([0-9]+)\s*/\s*([0-9]+)\b")
_unit_re = re.compile(
    r"\b([0-9][0-9.,]*)\s*(kg|g|km|m|cm|mm|m2|m²|km2|km²|gb|mb|kb|tb|hz|khz|mhz|ghz|ml|l)\b",
    re.IGNORECASE,
)
_decimal_re = re.compile(r"(-?[0-9]+)[\.,]([0-9]+)")
_long_digit_re = re.compile(r"\b\d{8,15}\b")
_integer_re = re.compile(r"-?[0-9]+")

_ones = [
    "không", "một", "hai", "ba", "bốn", "năm",
    "sáu", "bảy", "tám", "chín",
]

_teens = [
    "mười", "mười một", "mười hai", "mười ba", "mười bốn", "mười lăm",
    "mười sáu", "mười bảy", "mười tám", "mười chín",
]

_UNIT_MAP = {
    "kg": "ki lô gam",
    "g": "gam",
    "km": "ki lô mét",
    "m": "mét",
    "cm": "xen ti mét",
    "mm": "mi li mét",
    "m2": "mét vuông",
    "m²": "mét vuông",
    "km2": "ki lô mét vuông",
    "km²": "ki lô mét vuông",
    "gb": "ghi ga bai",
    "mb": "mê ga bai",
    "kb": "ki lô bai",
    "tb": "tê ra bai",
    "hz": "héc",
    "khz": "ki lô héc",
    "mhz": "mê ga héc",
    "ghz": "ghi ga héc",
    "ml": "mi li lít",
    "l": "lít",
}

_CURRENCY_MAP = {
    "đ": "đồng",
    "vnđ": "đồng",
    "vnd": "đồng",
    "usd": "đô la mỹ",
    "eur": "ơ rô",
    "gbp": "bảng anh",
    "jpy": "yên nhật",
}

def _say_two_digits(n: int, is_leading: bool = False) -> str:
    assert 0 <= n <= 99
    if n < 10:
        if is_leading:
            return _ones[n]
        return "lẻ " + _ones[n] if n > 0 else "không trăm"
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

def _int_to_words(n: int) -> str:
    if n < 10:
        return _ones[n]
    if n < 20:
        return _teens[n - 10]
    if n < 100:
        return _say_two_digits(n, is_leading=True)
    if n < 1000:
        tram = n // 100
        remainder = n % 100
        text = _ones[tram] + " trăm"
        if remainder == 0:
            return text
        return text + " " + _say_two_digits(remainder)
    if n < 1_000_000:
        nghin = n // 1000
        remainder = n % 1000
        text = _int_to_words(nghin) + " nghìn"
        if remainder == 0:
            return text
        if remainder < 100:
            return text + " không trăm " + _say_two_digits(remainder)
        return text + " " + _int_to_words(remainder)
    if n < 1_000_000_000:
        trieu = n // 1_000_000
        remainder = n % 1_000_000
        text = _int_to_words(trieu) + " triệu"
        if remainder == 0:
            return text
        return text + " " + _int_to_words(remainder)
    ty = n // 1_000_000_000
    remainder = n % 1_000_000_000
    text = _int_to_words(ty) + " tỷ"
    if remainder == 0:
        return text
    return text + " " + _int_to_words(remainder)

def _normalize_number_string(number_str: str) -> str:
    number_str = number_str.strip().replace(" ", "")

    if "," in number_str and "." in number_str:
        if number_str.rfind(",") > number_str.rfind("."):
            number_str = number_str.replace(".", "").replace(",", ".")
        else:
            number_str = number_str.replace(",", "")
    elif number_str.count(".") > 1:
        number_str = number_str.replace(".", "")
    elif number_str.count(",") > 1:
        number_str = number_str.replace(",", "")
    elif "," in number_str:
        left, right = number_str.split(",", 1)
        if len(right) == 3 and left.isdigit():
            number_str = left + right
        else:
            number_str = left + "." + right
    elif "." in number_str:
        left, right = number_str.split(".", 1)
        if len(right) == 3 and left.isdigit():
            number_str = left + right

    return number_str

def _number_string_to_words(number_str: str) -> str:
    normalized = _normalize_number_string(number_str)

    if "." in normalized:
        integer_part, frac_part = normalized.split(".", 1)
        sign = ""
        if integer_part.startswith("-"):
            sign = "âm "
            integer_part = integer_part[1:]
        int_words = _int_to_words(int(integer_part or "0"))
        frac_words = " ".join(_ones[int(d)] for d in frac_part)
        return sign + int_words + " phẩy " + frac_words

    n = int(normalized)
    if n < 0:
        return "âm " + _int_to_words(-n)
    return _int_to_words(n)

def _digit_by_digit(number_str: str) -> str:
    return " ".join(_ones[int(ch)] for ch in number_str if ch.isdigit())

def _expand_date(match: "re.Match") -> str:
    day = int(match.group(1))
    month = int(match.group(2))
    year = int(match.group(3))
    return f"ngày {_int_to_words(day)} tháng {_int_to_words(month)} năm {_int_to_words(year)}"

def _expand_year_range(match: "re.Match") -> str:
    start_year = int(match.group(1))
    end_year = int(match.group(2))
    return f"từ năm {_int_to_words(start_year)} đến năm {_int_to_words(end_year)}"

def _expand_phone(match: "re.Match") -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return _digit_by_digit(digits)

def _expand_percent(match: "re.Match") -> str:
    return _number_string_to_words(match.group(1)) + " phần trăm"

def _expand_currency(match: "re.Match") -> str:
    amount = _number_string_to_words(match.group(1))
    currency = _CURRENCY_MAP[match.group(2).lower()]
    return f"{amount} {currency}"

def _expand_fraction(match: "re.Match") -> str:
    numerator = int(match.group(1))
    denominator = int(match.group(2))
    if numerator == 1 and denominator == 2:
        return "một phần hai"
    if numerator == 1 and denominator == 3:
        return "một phần ba"
    if numerator == 1 and denominator == 4:
        return "một phần tư"
    return f"{_int_to_words(numerator)} phần {_int_to_words(denominator)}"

def _expand_unit(match: "re.Match") -> str:
    amount = _number_string_to_words(match.group(1))
    unit = _UNIT_MAP[match.group(2).lower()]
    return f"{amount} {unit}"

def _expand_decimal(match: "re.Match") -> str:
    return _number_string_to_words(match.group(0))

def _expand_long_digits(match: "re.Match") -> str:
    return _digit_by_digit(match.group(0))

def _expand_integer(match: "re.Match") -> str:
    return _number_string_to_words(match.group(0))

def normalize_numbers(text: str) -> str:
    text = re.sub(_date_re, _expand_date, text)
    text = re.sub(_year_range_re, _expand_year_range, text)
    text = re.sub(_phone_re, _expand_phone, text)
    text = re.sub(_percent_re, _expand_percent, text)
    text = re.sub(_currency_re, _expand_currency, text)
    text = re.sub(_fraction_re, _expand_fraction, text)
    text = re.sub(_unit_re, _expand_unit, text)
    text = re.sub(_decimal_re, _expand_decimal, text)
    text = re.sub(_long_digit_re, _expand_long_digits, text)
    text = re.sub(_integer_re, _expand_integer, text)
    return text