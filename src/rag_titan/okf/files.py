"""
List the files of an OKF bundle, from a folder or a .zip, safely (OKF v0.3, 3.2).

Files and folders whose names start with "." are not part of a bundle. A
symbolic link that leads out of the bundle root is never followed, and a zip
entry that would land outside the bundle (an absolute path, "..") is skipped,
so reading a bundle can never read other files on your computer.
"""

import os
import zipfile
from pathlib import Path, PurePosixPath


def problem(level, path, message):
    """One problem found while reading: level is "error" or "warning"."""
    return {"level": level, "path": path, "message": message}


def is_hidden(parts):
    """True when any part of the path starts with "." (not part of the bundle)."""
    return any(part.startswith(".") for part in parts)


def folder_files(root):
    """
    (path in the bundle, bytes) for every file under `root`, and the problems.

    Paths use "/" on every system, for example "metrics/revenue.md".
    """
    root = Path(root).resolve()
    files, problems = [], []
    for folder, subfolders, names in os.walk(root, followlinks=False):
        subfolders[:] = sorted(d for d in subfolders if not d.startswith("."))
        for name in sorted(names):
            full = Path(folder) / name
            relative = full.relative_to(root).as_posix()
            if is_hidden(PurePosixPath(relative).parts):
                continue
            if full.is_symlink() and not is_inside(full.resolve(), root):
                problems.append(problem("warning", relative,
                                        "a symbolic link out of the bundle: not read"))
                continue
            if full.is_file():
                files.append((relative, full.read_bytes()))
    return files, problems


def zip_files(path):
    """
    (path in the bundle, bytes) for every file in a zip, and the problems.

    A zip of a folder usually holds one top folder ("my_bundle/..."): then that
    folder is the bundle root.
    """
    files, problems = [], []
    with zipfile.ZipFile(path) as archive:
        entries = [info for info in archive.infolist() if not info.is_dir()]
        names = [PurePosixPath(info.filename.replace("\\", "/")) for info in entries]
        tops = {name.parts[0] for name in names if len(name.parts) > 1}
        strip = len(tops) == 1 and all(len(name.parts) > 1 for name in names)
        for info, name in zip(entries, names):
            if name.is_absolute() or ".." in name.parts:
                problems.append(problem("warning", info.filename,
                                        "a zip entry outside the bundle: not read"))
                continue
            parts = name.parts[1:] if strip else name.parts
            if parts and not is_hidden(parts):
                files.append(("/".join(parts), archive.read(info)))
    return sorted(files), problems


def is_inside(path, root):
    """True when `path` is `root` or lies under it."""
    try:
        Path(path).relative_to(root)
    except ValueError:
        return False
    return True


def bundle_files(source):
    """
    The files of a bundle at `source`: a folder or a .zip file.

    Raises:
        ValueError: `source` is neither.
    """
    source = Path(source)
    if source.is_dir():
        return folder_files(source)
    if source.is_file() and zipfile.is_zipfile(source):
        return zip_files(source)
    raise ValueError(f"An OKF bundle is a folder or a .zip file: {str(source)!r} is neither")
