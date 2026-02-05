from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import csv
import json
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_dump(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_write(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    p = Path(path)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

