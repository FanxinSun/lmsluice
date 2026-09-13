"""Deterministic evidence archive creation for the user-operated probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import zipfile


def _files(run_dir: str, archive_path: str) -> list[tuple[str, str]]:
    root = os.path.abspath(run_dir)
    archive = os.path.abspath(archive_path)
    found = []
    for directory, _names, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(directory, name)
            if os.path.abspath(path) == archive:
                continue
            # Failure campaigns may intentionally leave a FIFO or symlink in
            # their retained fixture tree. Opening one while creating the
            # evidence archive could block forever or copy data outside the
            # run. Structured failure rows already retain the detection; the
            # archive includes only regular evidence files.
            try:
                mode = os.lstat(path).st_mode
            except OSError:
                continue
            if not stat.S_ISREG(mode):
                continue
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            found.append((relative, path))
    return sorted(found)


def create_archive(run_dir: str, archive_path: str) -> dict:
    """Write a stable ZIP member order, timestamp and permission set."""
    os.makedirs(os.path.dirname(os.path.abspath(archive_path)), exist_ok=True)
    members = _files(run_dir, archive_path)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for relative, path in members:
            with open(path, "rb") as fh:
                payload = fh.read()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.extra = b""
            info.comment = b""
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
    return inspect_archive(archive_path, required_members=members)


def inspect_archive(archive_path: str, *, required_members=None) -> dict:
    """List and read an evidence archive, returning size and digest facts."""
    path = os.path.abspath(archive_path)
    with zipfile.ZipFile(path, "r") as archive:
        names = sorted(archive.namelist())
        read_names = set(names)
        if required_members:
            required_names = {name for name, _path in required_members}
            missing = sorted(required_names - read_names)
        else:
            missing = []
        bad = archive.testzip()
        # Read every member so a truncated central directory or body is not
        # reported as usable merely because it can be listed.
        for name in names:
            archive.read(name)
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return {
        "path": path,
        "bytes": os.path.getsize(path),
        "sha256": digest.hexdigest(),
        "members": names,
        "member_count": len(names),
        "missing_members": missing,
        "bad_member": bad,
        "readable": bad is None and not missing,
        "format": "deterministic-zip",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        result = inspect_archive(args.archive)
    else:
        result = create_archive(args.run_dir, args.archive)
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("readable") else 3


if __name__ == "__main__":  # pragma: no cover - exercised by the shell probe
    raise SystemExit(main())
