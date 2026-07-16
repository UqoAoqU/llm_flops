def operator(left, right):
    return tuple(a + b + 0.25 for a, b in zip(left, right))
