"""
Sliding windows: the shared core of every fixed-length chunker.

A window holds `size` units (characters, words or token ids). Each new window
starts `size - overlap` units after the previous one.
Example: size 4, overlap 1 on "abcdefghij" -> "abcd", "defg", "ghij".
"""


def check_window_size(size, overlap):
    """
    Raise ValueError when size and overlap cannot make sliding windows.

    Args:
        size: number of units in each window, at least 1.
        overlap: units shared by two neighbouring windows, from 0 to size - 1.
    """
    if not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a whole number greater than 0")

    # overlap >= size would never move the window forward
    if not isinstance(overlap, int) or overlap < 0 or overlap >= size:
        raise ValueError("overlap must be a whole number between 0 and size - 1")


def fixed_windows(units, size, overlap=0):
    """
    Cut `units` into overlapping windows of `size` units each.

    Every window has exactly `size` units except the last one, which may be
    shorter. No window is empty and no unit is lost.

    Args:
        units: a string (characters) or a list (words, token ids, ...).
        size: number of units in each window.
        overlap: number of units shared by two neighbouring windows.

    Returns:
        List of slices of `units` (strings for a string, lists for a list).
    """
    check_window_size(size, overlap)
    step = size - overlap
    windows = []

    for start in range(0, len(units), step):
        end = start + size
        windows.append(units[start:end])

        # stop once this window reached the end, so the tail is not repeated
        if end >= len(units):
            break

    return windows
