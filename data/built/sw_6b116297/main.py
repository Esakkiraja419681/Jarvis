"""Hello, Sir."""


SPEC = 'hello world greeting program'


def main() -> int:
    print(f"Hello, Sir. Working on: {SPEC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
