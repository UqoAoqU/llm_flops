import sys

sys.stdout.write("O" * (1024 * 1024))
sys.stdout.flush()
sys.stderr.write("E" * (1024 * 1024))
sys.stderr.flush()


def operator(value: object) -> object:
    return value
