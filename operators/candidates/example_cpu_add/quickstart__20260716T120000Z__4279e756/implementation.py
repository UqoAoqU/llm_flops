def operator(left, right):
    return tuple(a + b for a, b in zip(left, right))
