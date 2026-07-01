import re

abbreviations_vi = [
    (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
    for x in [
        ("tp", "thành phố"),
        ("ts", "tiến sĩ"),
        ("ths", "thạc sĩ"),
        ("gs", "giáo sư"),
        ("pgs", "phó giáo sư"),
        ("q", "quận"),
        ("p", "phường"),
        ("tt", "thị trấn"),
        ("tx", "thị xã"),
        ("v\\.v", "vân vân"),
        ("ubnd", "ủy ban nhân dân"),
        ("thpt", "trung học phổ thông"),
        ("thcs", "trung học cơ sở"),
    ]
]