from fastspeech2.text import cmudict, vietnamese_phonemes

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_vi  = 'aáảàãạâấẩầẫậăắẳằẵặbcdđeéẻèẽẹêếểềễệfghiíỉìĩịjklmnoóỏòõọôốổồỗộơớởờỡợpqrstuúủùũụưứửừữựvwxyýỷỳỹỵz'

_silences = ["@sp", "@spn", "@sil"]

_arpabet = ["@" + s for s in cmudict.valid_symbols]
_vietnamese_phonemes = ["@" + s for s in vietnamese_phonemes.valid_symbols]

symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _silences
)

symbols_vi = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters_vi)
    + _vietnamese_phonemes
    + _silences
)

def get_symbols(vi_lang=False):
    if vi_lang:
        return symbols_vi
    else:
        return symbols


import os as _os

_target_symbol_count = _os.environ.get("FASTSPEECH2_SYMBOL_COUNT")

if _target_symbol_count:
    _target_symbol_count = int(_target_symbol_count)

    if len(symbols) < _target_symbol_count:
        symbols = list(symbols) + [
            "@dummy_{}".format(i)
            for i in range(_target_symbol_count - len(symbols))
        ]

    elif len(symbols) > _target_symbol_count:
        raise RuntimeError(
            "Current symbols count {} > target symbols count {}".format(
                len(symbols), _target_symbol_count
            )
        )

