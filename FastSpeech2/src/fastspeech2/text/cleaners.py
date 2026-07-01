import re
from unidecode import unidecode
from .numbers import normalize_numbers
from . import numbers_vi as nvi

_whitespace_re = re.compile(r'\s+')
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]

_abbreviations_vi = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('btc', 'ban tổ chức'),
    ('clb', 'câu lạc bộ'),
    ('htx', 'hợp tác xã'),
    ('nxb', 'nhà xuất bản'),
    ('ôb', 'ông bà'),
    ('tp', 'thành phố'),
    ('tt', 'tổng thống'),
    ('ttg', 'thủ tướng'),
    ('tw', 'trung ương'),
    ('ubnd', 'ủy ban nhân dân'),
    ('bch', 'ban chấp hành'),
    ('chxhcnvn', 'cộng hòa xã hội chủ nghĩa việt nam'),
    ('mtdtgpmnvn', 'mặt trận dân tộc giải phóng miền nam việt nam'),
    ('qdnd', 'quân đội nhân dân việt nam'),
    ('qđnd', 'quân đội nhân dân việt nam'),
    ('vn', 'việt nam'),
    ('qlvnch', 'quân lực việt nam cộng hòa'),
    ('vnqdđ', 'việt nam quốc dân đảng'),
    ('vnqdd', 'việt nam quốc dân đảng'),
    ('vnch', 'việt nam cộng hòa'),
    ('vndcch', 'việt nam dân chủ cộng hòa'),
    ('lhq', 'liên Hợp quốc'),
    ('thpt', 'trung học phổ thông'),
    ('thcs', 'trung học cơ sở'),
    ('đ/c', 'địa chỉ'),
    ('k/g', 'kính gửi'),
    ('th/g', 'thân gửi'),
    ('v/v', 'về việc'),
    ('tr', 'trang'),
    ('dc', 'được'),
    ('đc', 'được'),
    ('cty', 'công ty'),
    ('ngta', 'người ta'),
    ('tv', 'ti vi'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_abbreviations_vi(text):
    for regex, replacement in _abbreviations_vi:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def vietnamese_cleaners(text):
    text = lowercase(text)
    text = nvi.normalize_numbers(text)
    text = expand_abbreviations_vi(text)
    text = collapse_whitespace(text)
    return text
