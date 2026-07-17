"""Windows terminal selection menu shared by CLI workflows."""

import sys


def select_menu(title: str, options, current=None, action_keys=None, hint=None):
    """Render a keyboard-driven selection menu and return the selected value."""
    import msvcrt

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

    def render():
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
    render()

    while True:
        key = msvcrt.getch()
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
        if key in (b"\xe0", b"\x00"):
            k2 = msvcrt.getch()
            if k2 == b"H":
                idx = (idx - 1) % len(normalized)
                render()
            elif k2 == b"P":
                idx = (idx + 1) % len(normalized)
                render()
        if key.isdigit():
            n = int(key)
            if 1 <= n <= len(normalized):
                idx = n - 1
                render()
