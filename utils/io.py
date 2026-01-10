import json
import os
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Sequence, Union


def write_jsonl(path: Union[str, Path], records: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def write_text(path: Union[str, Path], text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def read_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj


def read_json(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        # allow {"data":[...]} style
        if "data" in obj and isinstance(obj["data"], list):
            obj = obj["data"]
    for i, row in enumerate(obj):
        yield row


def read_csv(path: Path) -> Iterator[Dict[str, Any]]:
    dialect = "excel-tab" if path.suffix.lower() == ".tsv" else "excel"
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            yield row


def iter_upstream_records(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
) -> Iterator[Dict[str, Any]]:
    """
    Iterate upstream records from one or multiple files.
    Supports .jsonl or .json.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for p in paths:
        p = Path(p)
        suffix = p.suffix.lower()
        if suffix == ".jsonl":
            yield from read_jsonl(p)
        elif suffix == ".json":
            yield from read_json(p)
