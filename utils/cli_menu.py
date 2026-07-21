"""Cross-platform terminal selection menu shared by CLI workflows."""

import os
import sys


def _select_menu_windows(
    title: str,
    normalized: list[tuple[str, object]],
    idx: int,
    action_keys: dict | None,
    hint: str | None,
) -> object | None:
    import msvcrt

    def _getch() -> bytes:
        return msvcrt.getch()

    def _render() -> None:
        lines = len(normalized) + 3
        sys.stdout.write("\r" + "\033[A" * lines + "\033[J")
        print(f"\033[36m{title}\033[0m")
        print(hint or "  (\u2191\u2193 \u9009\u62e9, Enter \u786e\u8ba4, Esc \u53d6\u6d88)")
        for i, (label, _) in enumerate(normalized):
            if i == idx:
                print(f"  \033[32m\u2771 {label}\033[0m")
            else:
                print(f"    {label}")
        print("-" * 40)

    print()
    _render()

    while True:
        first = _getch()
        if first in (b"\xe0", b"\x00"):
            direction = _getch()
            if direction == b"H":
                idx = (idx - 1) % len(normalized)
                _render()
            elif direction == b"P":
                idx = (idx + 1) % len(normalized)
                _render()
        elif first in (b"\r", b"\n"):
            lines = len(normalized) + 3
            sys.stdout.write("\r" + "\033[A" * lines + "\033[J")
            label, value = normalized[idx]
            print(f"{title} \033[32m\u2771 {label}\033[0m")
            return value
        elif first == b"\x1b":
            lines = len(normalized) + 3
            sys.stdout.write("\r" + "\033[A" * lines + "\033[J")
            print(f"{title} \033[90m(\u5df2\u53d6\u6d88)\033[0m")
            return None
        elif action_keys and first in action_keys:
            lines = len(normalized) + 3
            sys.stdout.write("\r" + "\033[A" * lines + "\033[J")
            label, value = normalized[idx]
            action = action_keys[first]
            print(f"{title} \033[33m{action}: {label}\033[0m")
            return (action, value)
        elif first.isdigit():
            n = int(first)
            if 1 <= n <= len(normalized):
                idx = n - 1
                _render()


def _select_menu_unix(
    title: str,
    normalized: list[tuple[str, object]],
    idx: int,
    action_keys: dict | None,
    hint: str | None,
) -> object | None:
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        def _getch() -> bytes:
            return os.read(fd, 1)

        def _println(text: str = "") -> None:
            sys.stdout.write(text + "\r\n")

        def _read_key() -> bytes | tuple[bytes, bytes]:
            first = _getch()
            if first == b"\x1b":
                r, _, _ = select.select([fd], [], [], 0.05)
                if r:
                    second = _getch()
                    if second == b"[":
                        r2, _, _ = select.select([fd], [], [], 0.05)
                        if r2:
                            third = _getch()
                            return (b"\x1b", third)
                        return b"\x1b"
                    if second == b"O":
                        r2, _, _ = select.select([fd], [], [], 0.05)
                        if r2:
                            third = _getch()
                            return (b"\x1b", third)
                        return b"\x1b"
                return b"\x1b"
            return first

        def _render() -> None:
            lines = len(normalized) + 3
            sys.stdout.write("\r" + "\033[A" * lines + "\033[J")
            _println(f"\033[36m{title}\033[0m")
            _println(hint or "  (\u2191\u2193 \u9009\u62e9, Enter \u786e\u8ba4, Esc \u53d6\u6d88)")
            for i, (label, _) in enumerate(normalized):
                if i == idx:
                    _println(f"  \033[32m\u2771 {label}\033[0m")
                else:
                    _println(f"    {label}")
            _println("-" * 40)
            sys.stdout.flush()

        _println()
        _render()

        while True:
            key = _read_key()

            if isinstance(key, bytes):
                if key in (b"\r", b"\n"):
                    lines = len(normalized) + 3
                    sys.stdout.write("\r" + "\033[A" * lines + "\033[J")
                    label, value = normalized[idx]
                    _println(f"{title} \033[32m\u2771 {label}\033[0m")
                    return value
                if key == b"\x1b":
                    lines = len(normalized) + 3
                    sys.stdout.write("\r" + "\033[A" * lines + "\033[J")
                    _println(f"{title} \033[90m(\u5df2\u53d6\u6d88)\033[0m")
                    return None
                if action_keys and key in action_keys:
                    lines = len(normalized) + 3
                    sys.stdout.write("\r" + "\033[A" * lines + "\033[J")
                    label, value = normalized[idx]
                    action = action_keys[key]
                    _println(f"{title} \033[33m{action}: {label}\033[0m")
                    return (action, value)
                if key.isdigit():
                    n = int(key)
                    if 1 <= n <= len(normalized):
                        idx = n - 1
                        _render()
                continue

            _prefix, direction = key
            if direction in (b"A", b"k"):
                idx = (idx - 1) % len(normalized)
                _render()
            elif direction in (b"B", b"j"):
                idx = (idx + 1) % len(normalized)
                _render()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def select_menu(title, options, current=None, action_keys=None, hint=None):
    """Render a keyboard-driven selection menu and return the selected value."""
    normalized = []
    for opt in options:
        if isinstance(opt, (tuple, list)) and len(opt) == 2:
            normalized.append((str(opt[0]), opt[1]))
        else:
            normalized.append((str(opt), opt))

    if not normalized:
        return None

    idx = 0
    if current is not None:
        for i, (label, value) in enumerate(normalized):
            if value == current or label == str(current):
                idx = i
                break

    if sys.platform == "win32":
        return _select_menu_windows(title, normalized, idx, action_keys, hint)
    return _select_menu_unix(title, normalized, idx, action_keys, hint)
