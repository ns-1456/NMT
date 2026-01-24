#!/usr/bin/env python3
"""
XLCoST snippet-level code translation data preparation (Python -> C++).

This script:
- Downloads the official XLCoST release zip from Google Drive (via gdown)
- Extracts it under data/raw/
- Reads snippet-level parallel files for the Python<->C++ pair
- Prefixes Python source with: "translate Python to C++: "
- Writes:
  - data/processed/corpus.txt (for tokenizer training)
  - Arrow dataset (datasets.DatasetDict) via save_to_disk()
  - train/validation/test JSONL exports

Important: we do NOT strip whitespace from code lines (beyond removing the
line-ending newline when reading lines from files).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Literal


XL_COST_GDRIVE_ID = "1Cp3vFITRaUEJwPoeI_uv0cC6KVyvDc4F"
DEFAULT_PREFIX = "translate Python to C++: "
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_DATASET_DIRNAME = "xlcost_py_cpp_snippet"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _download_xlcost_zip(zip_path: Path, gdrive_id: str) -> None:
    _ensure_dir(zip_path.parent)
    if zip_path.exists() and zip_path.stat().st_size > 0:
        print(f"[data_prep] Found existing zip at {zip_path}")
        return

    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print(f"[data_prep] Downloading XLCoST zip to {zip_path}")
    # Prefer gdown if available (handles Google Drive confirm tokens).
    try:
        import gdown  # type: ignore

        out = gdown.download(url, str(zip_path), quiet=False)
        if out is None:
            raise RuntimeError("gdown returned None")
        if not zip_path.exists() or zip_path.stat().st_size == 0:
            raise RuntimeError("zip missing or empty after gdown download")
        return
    except ModuleNotFoundError:
        print("[data_prep] gdown not installed; using a minimal Google Drive downloader")

    _download_gdrive_file_minimal(file_id=gdrive_id, out_path=zip_path)


def _download_gdrive_file_minimal(file_id: str, out_path: Path) -> None:
    """
    Minimal Google Drive downloader without third-party deps.
    This handles the common \"download_warning\" confirm-token flow.
    """
    import re
    import urllib.request
    from http.cookiejar import CookieJar

    _ensure_dir(out_path.parent)

    base_url = "https://drive.google.com/uc?export=download"
    jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    def fetch(url: str) -> bytes:
        with opener.open(url) as resp:
            return resp.read()

    # First request: may yield HTML requiring confirmation for large files
    url1 = f"{base_url}&id={file_id}"
    data = fetch(url1)

    # Look for a confirm token in cookies or HTML
    confirm = None
    for c in jar:
        if c.name.startswith("download_warning"):
            confirm = c.value
            break

    if confirm is None:
        # Sometimes token is embedded in HTML
        m = re.search(rb"confirm=([0-9A-Za-z_-]+)", data)
        if m:
            confirm = m.group(1).decode("utf-8")

    if confirm is not None:
        url2 = f"{base_url}&confirm={confirm}&id={file_id}"
    else:
        # If no confirm token, the first response might already be the file.
        url2 = url1

    print(f"[data_prep] Downloading from Google Drive to {out_path}")
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    with opener.open(url2) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    tmp.rename(out_path)
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("Download failed (zip not found or empty).")


def _extract_zip(zip_path: Path, raw_dir: Path) -> None:
    # The zip contains an `XLCoST_data/` folder (possibly nested).
    print(f"[data_prep] Extracting {zip_path} into {raw_dir}")
    _ensure_dir(raw_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)


def _find_xlcost_root(raw_dir: Path) -> Path:
    """
    Return path to the extracted XLCoST_data directory.
    We avoid a full os.walk (the extraction can be large); instead, try common
    locations and a shallow scan.
    """
    direct = raw_dir / "XLCoST_data"
    if direct.is_dir():
        return direct

    # Sometimes the zip contains a parent folder, e.g. raw_dir/<something>/XLCoST_data
    for child in raw_dir.iterdir():
        if not child.is_dir():
            continue
        candidate = child / "XLCoST_data"
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not find extracted XLCoST_data directory under {raw_dir}"
    )


def _pair_dir(xlcost_root: Path, level: Literal["snippet", "program"]) -> Path:
    if level == "snippet":
        base = xlcost_root / "pair_data_tok_1"
    else:
        base = xlcost_root / "pair_data_tok_full"
    if not base.is_dir():
        raise FileNotFoundError(f"Missing expected directory: {base}")
    return base


def _detect_lang_pair_dir(
    base: Path, preferred_dirs: Iterable[str]
) -> Path:
    for name in preferred_dirs:
        cand = base / name
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        f"Could not find any of these language pair dirs under {base}: "
        + ", ".join(preferred_dirs)
    )


def _choose_split_file(
    pair_dir: Path,
    split: str,
    ext_candidates: list[str],
) -> Path:
    """
    Find the split file for a language based on extension candidates.
    Example file patterns from XLCoST:
      train-C++-C-tok.cpp
      train-C++-Python-tok.py
    """
    files = [p.name for p in pair_dir.iterdir() if p.is_file()]
    # Prefer "-tok" files (tokenized) and avoid map.jsonl
    for ext in ext_candidates:
        matches = [
            f
            for f in files
            if f.startswith(f"{split}-")
            and "-tok" in f
            and f.endswith(ext)
            and "map.jsonl" not in f
        ]
        if len(matches) == 1:
            return pair_dir / matches[0]
        if len(matches) > 1:
            # Pick the shortest name (usually the canonical one).
            matches.sort(key=len)
            return pair_dir / matches[0]

    # Fallback: pick any split file by extension, even if "-tok" missing.
    for ext in ext_candidates:
        matches = [
            f
            for f in files
            if f.startswith(f"{split}-")
            and f.endswith(ext)
            and "map.jsonl" not in f
        ]
        if len(matches) == 1:
            return pair_dir / matches[0]
        if len(matches) > 1:
            matches.sort(key=len)
            return pair_dir / matches[0]

    raise FileNotFoundError(
        f"Could not find split='{split}' file in {pair_dir} "
        f"with extensions {ext_candidates}. Present files: {sorted(files)[:25]}..."
    )


def _read_lines_keep_ws(path: Path) -> list[str]:
    # Do not strip whitespace. splitlines() removes only the line-ending newlines.
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def processed_dataset_dir(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> Path:
    return processed_dir / DEFAULT_DATASET_DIRNAME


def load_processed_dataset(processed_dir: Path = DEFAULT_PROCESSED_DIR):
    """
    Load the processed DatasetDict written by this script (save_to_disk).
    """
    from datasets import load_from_disk  # type: ignore

    path = processed_dataset_dir(processed_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. "
            "Run data_prep.py with datasets installed to create it."
        )
    return load_from_disk(str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--gdrive_id", type=str, default=XL_COST_GDRIVE_ID)
    ap.add_argument("--level", choices=["snippet", "program"], default="snippet")
    ap.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip_download",
        action="store_true",
        help="Assume XLCoST zip is already present under raw_dir.",
    )
    ap.add_argument(
        "--skip_extract",
        action="store_true",
        help="Assume XLCoST_data has already been extracted under raw_dir.",
    )
    args = ap.parse_args()

    raw_dir: Path = args.raw_dir
    processed_dir: Path = args.processed_dir
    _ensure_dir(raw_dir)
    _ensure_dir(processed_dir)

    zip_path = raw_dir / "XLCoST_data.zip"

    if not args.skip_download:
        _download_xlcost_zip(zip_path=zip_path, gdrive_id=args.gdrive_id)
    else:
        if zip_path.exists():
            print(f"[data_prep] Skipping download; using {zip_path}")
        else:
            # Allow skipping download if extraction already exists (useful for local smoke tests).
            try:
                xlcost_root = _find_xlcost_root(raw_dir)
                print(
                    f"[data_prep] Skipping download; zip missing but found extracted dir at {xlcost_root}"
                )
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"--skip_download set but neither zip ({zip_path}) nor extracted XLCoST_data exists"
                ) from e

    if not args.skip_extract:
        # If already extracted, skip re-extract to avoid wasting time.
        try:
            xlcost_root = _find_xlcost_root(raw_dir)
            print(f"[data_prep] Found existing extracted dir at {xlcost_root}; skipping extract")
        except FileNotFoundError:
            _extract_zip(zip_path=zip_path, raw_dir=raw_dir)
    else:
        print("[data_prep] Skipping extract")

    xlcost_root = _find_xlcost_root(raw_dir)
    base = _pair_dir(xlcost_root, level=args.level)

    # Directory naming in XLCoST uses "C++-Python" etc.
    pair_dir = _detect_lang_pair_dir(
        base,
        preferred_dirs=["C++-Python", "Python-C++"],
    )
    print(f"[data_prep] Using pair directory: {pair_dir}")

    split_name_map = {"train": "train", "val": "validation", "test": "test"}

    # Determine which file is Python and which is C++ based on extension.
    # For Python->C++, we want python as source, cpp as target.
    split_rows: dict[str, list[dict[str, str]]] = {}
    rng = random.Random(args.seed)

    for split in ["train", "val", "test"]:
        cpp_file = _choose_split_file(pair_dir, split=split, ext_candidates=[".cpp"])
        py_file = _choose_split_file(pair_dir, split=split, ext_candidates=[".py", ".python"])

        py_lines = _read_lines_keep_ws(py_file)
        cpp_lines = _read_lines_keep_ws(cpp_file)

        if len(py_lines) != len(cpp_lines):
            raise ValueError(
                f"Line count mismatch for split={split}: "
                f"{py_file.name} has {len(py_lines)} lines, "
                f"{cpp_file.name} has {len(cpp_lines)} lines."
            )

        # Drop fully-empty aligned pairs (common if files end with a blank line).
        paired = list(zip(py_lines, cpp_lines))
        before = len(paired)
        paired = [(p, c) for (p, c) in paired if not (p == "" and c == "")]
        dropped = before - len(paired)
        if dropped:
            print(f"[data_prep] {split_name_map[split]}: dropped {dropped} fully-empty pairs")

        indices = list(range(len(paired)))
        if args.max_samples is not None and args.max_samples < len(indices):
            rng.shuffle(indices)
            indices = sorted(indices[: args.max_samples])

        rows = []
        for i in indices:
            py_line, cpp_line = paired[i]
            src = f"{args.prefix}{py_line}"
            tgt = cpp_line
            rows.append({"source": src, "target": tgt})

        split_rows[split_name_map[split]] = rows
        print(
            f"[data_prep] {split_name_map[split]}: {len(rows)} examples "
            f"(from {len(py_lines)} total)"
        )

    # Write tokenizer corpus
    corpus_path = processed_dir / "corpus.txt"
    print(f"[data_prep] Writing tokenizer corpus to {corpus_path}")
    with corpus_path.open("w", encoding="utf-8") as f:
        for split in ["train", "validation", "test"]:
            for r in split_rows[split]:
                f.write(r["source"] + "\n")
                f.write(r["target"] + "\n")

    # Save HF DatasetDict + JSONL exports
    out_ds_dir = processed_dir / "xlcost_py_cpp_snippet"
    try:
        from datasets import Dataset, DatasetDict  # type: ignore

        ds = DatasetDict(
            {
                "train": Dataset.from_list(split_rows["train"]),
                "validation": Dataset.from_list(split_rows["validation"]),
                "test": Dataset.from_list(split_rows["test"]),
            }
        )

        print(f"[data_prep] Saving Arrow dataset to {out_ds_dir}")
        if out_ds_dir.exists():
            # Remove to avoid datasets complaining about non-empty dir
            shutil.rmtree(out_ds_dir)
        ds.save_to_disk(str(out_ds_dir))
    except ModuleNotFoundError:
        print(
            "[data_prep] datasets not installed; skipping Arrow dataset save_to_disk(). "
            "JSONL exports are still written."
        )

    print("[data_prep] Writing JSONL exports")
    _write_jsonl(processed_dir / "train.jsonl", split_rows["train"])
    _write_jsonl(processed_dir / "validation.jsonl", split_rows["validation"])
    _write_jsonl(processed_dir / "test.jsonl", split_rows["test"])

    # Quick sample log
    sample = split_rows["train"][0] if split_rows["train"] else None
    if sample:
        print("[data_prep] Sample (train[0]):")
        print(f"  source_prefix_ok: {sample['source'].startswith(args.prefix)}")
        print(f"  source_len: {len(sample['source'])}  target_len: {len(sample['target'])}")

    print("[data_prep] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

