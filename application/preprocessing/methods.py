def one_hot(value: str, entries: list[tuple[str, int]]) -> list[float]:
    vect = len(entries) * [0.0]
    try:
        one_hot_idx: int = next(i for i, v in enumerate(entries)
                                if value == v[0])
    except StopIteration:
        one_hot_idx = 0
    vect[one_hot_idx] = 1.0
    return vect


def get_raw(value: str, entries: list[tuple[str, int]]) -> list[float]:
    try:
        val: int = next(v for i, v in enumerate(entries) if value == i)
    except StopIteration:
        val = -1
    return [val * 1.0]


def raw(value: float) -> list[float]:
    return [value]


# Standardise value v given average and standard deviation
# Most likely not usable because we have no idea what avg and sd of our dataset are
def standard(value: float, std: tuple[float, float]) -> float:
    return [(value - std[0]) / std[1]]
