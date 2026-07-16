import time


def operator(left, right):
    time.sleep(5)
    return tuple(a + b for a, b in zip(left, right))
