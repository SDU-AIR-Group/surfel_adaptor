import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar
import numpy as np

try:
    from tqdm import tqdm as tqdm  # type: ignore
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        """Small stderr progress fallback for minimal Python environments."""
        def __init__(self, iterable=None, *, total=None, initial=0, desc="Progress", unit="item", **_kwargs):
            self.iterable = iterable
            self.total = total if total is not None else (len(iterable) if iterable is not None else None)
            self.count = initial
            self.desc = desc
            self.unit = unit
            self._last = 0.0
            self._emit(force=True)

        def _emit(self, force=False):
            now = time.monotonic()
            if force or now - self._last >= 0.2:
                total = "?" if self.total is None else str(self.total)
                print(f"\\r{self.desc}: {self.count}/{total} {self.unit}", end="", file=sys.stderr, flush=True)
                self._last = now

        def update(self, value=1):
            self.count += value
            self._emit()

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            for value in self.iterable:
                yield value
                self.update(1)

        def close(self):
            self._emit(force=True)
            print(file=sys.stderr, flush=True)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()
            return False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_TOOLKIT = PROJECT_ROOT / "dataset_toolkit"


def default_blender() -> str:
    """Locate the project Blender build, with an environment override."""
    override = os.environ.get("BLENDER_BIN")
    if override:
        return override
    name = "blender-4.2.12-linux-x64/blender"
    candidates = (PROJECT_ROOT / name, PROJECT_ROOT.parent / name)
    return str(next((path for path in candidates if path.is_file()), candidates[0]))


DEFAULT_BLENDER = default_blender()
Task = TypeVar("Task")
Result = TypeVar("Result")

# Kept deliberately compatible with the existing terrain metadata files.
METADATA_HEADER = [
    "terrain_source", "terrain_tag",
    "patch_file", "patch_id",
    "scan_id", "scan_path",
    "feat_id", "feat_path",
    "render_subdir_id", "render_dir",
    "full_feat_id", "full_feat_path",
    "latent_id", "latent_path",
]


def add_common_arguments(parser: argparse.ArgumentParser, *, seed: bool = True) -> None:
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                        help="Dataset output directory.")
    parser.add_argument("--blender", default=DEFAULT_BLENDER,
                        help="Blender executable for headless jobs.")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter used for external voxel/feature/render adapters.")
    parser.add_argument("--toolkit-root", type=Path, default=DEFAULT_TOOLKIT,
                        help="Directory containing this dataset toolkit.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild outputs selected by this stage.")
    parser.add_argument("--dry-run", action="store_true", help="List work without launching tools or writing data.")
    parser.add_argument("--sync-every", type=int, default=25, help="Merge journal into metadata every N successes.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Defaults to <dataset-root>/logs/<stage>.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent independent jobs (default: 1; raise carefully for Blender/GPU stages).")
    if seed:
        parser.add_argument("--seed", type=int, default=0, help="Root seed for reproducible stochastic stages.")

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]

def sphere_segment_hammersley_sequence(
    n,
    num_samples,
    yaw_range=(0.0, 2 * np.pi),
    pitch_range=(-np.pi / 2, np.pi / 2),
    radius_range=None,
    offset=(0.0, 0.0, 0.0),
):
    u, v, w = hammersley_sequence(3, n, num_samples)
    u = (u + offset[0] / num_samples) % 1.0
    v = (v + offset[1]) % 1.0
    w = (w + offset[2]) % 1.0

    yaw_min, yaw_max = yaw_range
    pitch_min, pitch_max = pitch_range
    if yaw_max < yaw_min:
        yaw_max += 2 * np.pi
    if pitch_max < pitch_min:
        pitch_max += 2 * np.pi

    sin_min = np.sin(pitch_min)
    sin_max = np.sin(pitch_max)
    sin_theta = sin_min + (sin_max - sin_min) * u
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    pitch = np.arcsin(sin_theta)
    yaw = yaw_min + (yaw_max - yaw_min) * v

    if radius_range is None:
        radius = 1.0
    else:
        r_min, r_max = radius_range
        if r_min is None and r_max is None:
            radius = 1.0
        else:
            if r_min is None:
                r_min = r_max
            if r_max is None:
                r_max = r_min
            if r_max < r_min:
                r_min, r_max = r_max, r_min
            radius = r_min + (r_max - r_min) * w

    return [yaw, pitch, radius]

def ensure_layout(root: Path) -> None:
    for name in ("records", "logs", "temp", "scans", "voxels", "full_features", "renders", "latents"):
        (root / name).mkdir(parents=True, exist_ok=True)


def metadata_path(root: Path) -> Path:
    return root / "metadata.csv"


def journal_path(root: Path, stage: str) -> Path:
    return root / "records" / f"{stage}.jsonl"


def _normalize_row(row: dict[str, Any]) -> dict[str, str]:
    return {key: "" if row.get(key) is None else str(row.get(key, "")) for key in METADATA_HEADER}


def read_metadata(root: Path) -> list[dict[str, str]]:
    path = metadata_path(root)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        # Missing columns are upgraded on the next sync.
        return [_normalize_row(row) for row in reader]


def write_metadata(root: Path, rows: Iterable[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    target = metadata_path(root)
    fd, raw_tmp = tempfile.mkstemp(prefix="metadata.", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METADATA_HEADER)
            writer.writeheader()
            writer.writerows(_normalize_row(row) for row in rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(raw_tmp, target)
    finally:
        if os.path.exists(raw_tmp):
            os.unlink(raw_tmp)


def append_journal(root: Path, stage: str, record: dict[str, Any]) -> None:
    """Durably append an idempotent metadata operation after output is in place."""
    ensure_layout(root)
    path = journal_path(root, stage)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _apply_record(rows: list[dict[str, str]], record: dict[str, Any]) -> None:
    kind = record.get("kind")
    if kind == "new_scan":
        row = _normalize_row(record["row"])
        scan_id = row["scan_id"]
        for index, existing in enumerate(rows):
            if existing["scan_id"] == scan_id:
                rows[index] = {**existing, **row}
                return
        rows.append(row)
        return

    key = str(record.get("key", ""))
    # Ignore fields removed from the current schema.
    updates = {
        str(k): "" if v is None else str(v)
        for k, v in record.get("updates", {}).items()
        if str(k) in METADATA_HEADER
    }
    if kind == "scan":
        for row in rows:
            if row["scan_id"] == key:
                row.update(updates)
    elif kind == "patch":
        for row in rows:
            if row["patch_id"] == key:
                row.update(updates)
    else:
        raise ValueError(f"Unknown journal record kind: {kind!r}")


def reconcile_metadata(root: Path) -> list[dict[str, str]]:
    """Materialize all append-only journals into metadata.csv (operations are idempotent)."""
    ensure_layout(root)
    rows = read_metadata(root)
    for path in sorted((root / "records").glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    _apply_record(rows, json.loads(line))
                except Exception as exc:
                    raise ValueError(f"Invalid journal entry {path}:{line_number}: {exc}") from exc
    write_metadata(root, rows)
    return rows


def stable_token(*parts: object, length: int = 10) -> str:
    value = "|".join(str(part) for part in parts)
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def dataset_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def resolve_dataset_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def existing_file(root: Path, row: dict[str, str], path_column: str) -> bool:
    value = row.get(path_column, "")
    path = resolve_dataset_path(root, value) if value else None
    return bool(path and path.is_file() and path.stat().st_size > 0)


def existing_render(root: Path, row: dict[str, str]) -> bool:
    value = row.get("render_dir", "")
    transforms = resolve_dataset_path(root, value) / "transforms.json" if value else None
    return bool(transforms and transforms.is_file() and transforms.stat().st_size > 0)


def stage_log_dir(root: Path, stage: str, requested: Path | None) -> Path:
    path = requested if requested is not None else root / "logs" / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_command(command: list[str], log_path: Path, *, dry_run: bool = False) -> int:
    """Run a command while keeping its complete output in a per-item log."""
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        return subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False).returncode


def remove_for_overwrite(path: Path, overwrite: bool) -> None:
    if not overwrite or not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def patch_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Return the first scan row for each patch ID."""
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        patch_id = row.get("patch_id", "")
        if patch_id and patch_id not in result:
            result[patch_id] = row
    return result


def sync_due(success_count: int, every: int) -> bool:
    return every > 0 and success_count > 0 and success_count % every == 0


def parallel_jobs(tasks: Iterable[Task], worker: Callable[[Task], Result], *, workers: int,
                  desc: str, unit: str, initial: int = 0, total: int | None = None) -> Iterator[tuple[Result | None, Exception | None]]:
    """Run independent external-tool jobs; metadata remains owned by the caller."""
    if workers < 1:
        raise ValueError("--workers must be positive")
    items = list(tasks)
    if workers == 1:
        iterator = ((item, None) for item in items)
        for item, _ in tqdm(iterator, total=total if total is not None else len(items), initial=initial, desc=desc, unit=unit):
            try:
                yield worker(item), None
            except Exception as exc:
                yield None, exc
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dataset") as executor:
        futures = [executor.submit(worker, item) for item in items]
        for future in tqdm(as_completed(futures), total=total if total is not None else len(futures), initial=initial, desc=desc, unit=unit):
            try:
                yield future.result(), None
            except Exception as exc:
                yield None, exc


def dataset_counts(root: Path) -> dict[str, int]:
    """Count valid final artifacts without reading intermediate journals."""
    def files(path: Path, pattern: str) -> int:
        return sum(item.is_file() and item.stat().st_size > 0 for item in path.glob(pattern))

    patch_root = root / "1_terrain_splitting"
    render_root = root / "renders"
    return {
        "patches": sum(files(path / "patches", "patch_*.glb") for path in patch_root.iterdir())
        if patch_root.is_dir() else 0,
        "scans": files(root / "scans", "scan_*.npy"),
        "full_features": files(root / "full_features", "full_feat_*.npz"),
        "renders": sum((path / "transforms.json").is_file() for path in render_root.iterdir())
        if render_root.is_dir() else 0,
        "latents": files(root / "latents", "latent_*.npz"),
    }


def print_summary(root: Path, stage: str, *, processed: int, completed: int,
                  skipped: int, failed: int, failed_ids: Iterable[str] = ()) -> None:
    """Print the stage result and current dataset artifact totals."""
    counts = dataset_counts(root)
    print(
        f"\n[{stage}] processed={processed} succeeded={completed} "
        f"failed={failed} skipped={skipped}"
    )
    print(
        "[dataset] " + " ".join(
            f"{name}={counts[name]}" for name in
            ("patches", "scans", "full_features", "renders", "latents")
        )
    )
    ids = list(dict.fromkeys(failed_ids))
    if ids:
        print(f"[{stage}] failed_ids={','.join(ids)}")
