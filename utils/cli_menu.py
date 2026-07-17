"""Cross-platform terminal selection menu shared by CLI workflows."""

import sys


def _raw_getch() -> bytes:
    """Read a single raw byte from stdin (no echo, no line buffering)."""
    if sys.platform == "win32":
        import msvcrt

        return msvcrt.getch()
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.buffer.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key() -> bytes | tuple[bytes, bytes]:
    """Return a single key byte, or an (escape, action) pair for arrow / action keys."""
    first = _raw_getch()

    # On Unix, arrow keys are ESC [ A/B/C/D; other extended keys are ESC O or ESC [... .
    # On Windows, arrow keys are NUL/e0 followed by H/P/K/M.
    if first == b"\x1b" and sys.platform != "win32":
        import select

        # Peek: if a byte arrives within 50ms, it's part of an escape sequence.
        if select.select([sys.stdin], [], [], 0.05)[0]:
            second = _raw_getch()
            if second == b"[":
                third = _raw_getch()
                return (b"\x1b", third)
            if second == b"O":
                third = _raw_getch()
                return (b"\x1b", third)
        return b"\x1b"

    if first in (b"\xe0", b"\x00"):
        second = _raw_getch()
        return (first, second)

    return first


def select_menu(title: str, options, current=None, action_keys=None, hint=None):
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

    def _render() -> None:
        lines = len(normalized) + 3
        sys.stdout.write("\r" + "\033[A" * lines + "\033[J")
        print(f"\033[36m{title}\033[0m")
        print(hint or "  (↑↓ 选择, Enter 确认, Esc 取消)")
        for i, (label, _) in enumerate(normalized):
            if i == idx:
                print(f"  \033[32m❯ {label}\033[0m")
            else:
                print(f"    {label}")
        print("-" * 40)

    print()
    _render()

    while True:
        key = _read_key()

        # Single-byte keys
        if isinstance(key, bytes):
            if key in (b"\r", b"\n"):
                lines = len(normalized) + 3
                sys.stdout.write("\r" + "\033[A" * lines + "\033[J")
                label, value = normalized[idx]
                print(f"{title} \033[32m❯ {label}\033[0m")
                return value
            if key == b"\x1b":
                lines = len(normalized) + 3
                sys.stdout.write("\r" + "\033[A" * lines + "\033[J")
                print(f"{title} \033[90m(已取消)\033[0m")
                return None
            if action_keys and key in action_keys:
                lines = len(normalized) + 3
                sys.stdout.write("\r" + "\033[A" * lines + "\033[J")
                label, value = normalized[idx]
                action = action_keys[key]
                print(f"{title} \033[33m{action}: {label}\033[0m")
                return (action, value)
            if key.isdigit():
                n = int(key)
                if 1 <= n <= len(normalized):
                    idx = n - 1
                    _render()
            continue

        # Arrow / action key pair (prefix, direction)
        _prefix, direction = key
        if sys.platform == "win32":
            if direction == b"H":  # Up
                idx = (idx - 1) % len(normalized)
                _render()
            elif direction == b"P":  # Down
                idx = (idx + 1) % len(normalized)
                _render()
        else:
            if direction in (b"A", b"k"):  # Up / k
                idx = (idx - 1) % len(normalized)
                _render()
            elif direction in (b"B", b"j"):  # Down / j
                idx = (idx + 1) % len(normalized)
                _render()
