# FIXME: Bring back those commented countries
# because there are no students from those countries in filtered data,
# leading to incorrect one-hot vector.

nationality_inputs: list[tuple[str, int]] = [
    ('Portuguese', 1),
    ('German', 2),
    ('Spanish', 6),
    ('Italian', 11),
    ('Dutch', 13),
    ('English', 14),
    ('Lithuanian', 17),
    ('Angolan', 21),
    ('Cape Verdean', 22),
    ('Guinean', 24),
    ('Mozambican', 25),
    ('Santomean', 26),
    # ('Turkish', 32),
    ('Brazilian', 41),
    ('Romanian', 62),
    ('Moldova (Republic of)', 100),
    ('Mexican', 101),
    ('Ukrainian', 103),
    ('Russian', 105),
    # ('Cuban', 108),
    ('Colombian', 109),
]
