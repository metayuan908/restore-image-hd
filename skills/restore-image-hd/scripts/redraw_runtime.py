"""Durable, bounded generation jobs. No image calls occur at import or planning."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Journal:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def event(self, stage, **fields):
        row = {"time": utc_now(), "stage": stage, **fields}
        with self.lock, (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


class JobLock:
    """One coordinator per work directory; OS releases the lock after a crash."""
    def __init__(self, directory):
        self.path = Path(directory) / "coordinator.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0)
                self.stream.write("0")
                self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            self.stream.close()
            raise RuntimeError("Another coordinator owns this work directory.")
        return self

    def __exit__(self, *_):
        self.stream.close()


def is_rate_limit(error):
    message = str(error).lower()
    return "429" in message or "rate limit" in message or "too many requests" in message


def run_jobs(jobs, generate, inspect, directory, *, workers=2, resume=False,
             retries=1, retry_delay=2.0, reset_ids=()):
    """Generate immutable jobs, checkpoint each, serialize dispatch after a 429.

    generate(job, new_raw_path) owns its unique output. inspect(raw, job) must
    reject invalid native size/alpha. Ambiguous POST/network failures are NOT
    blindly retried; successful peers remain resumable.
    """
    if workers < 1 or workers > 4 or retries < 0:
        raise ValueError("workers must be 1..4; retries must be non-negative")
    directory = Path(directory)
    journal = Journal(directory)
    records, pending = {}, []
    started = time.monotonic()
    for job in jobs:
        folder = directory / "jobs" / job["id"] / job["signature"]
        folder.mkdir(parents=True, exist_ok=True)
        record_path = folder / "record.json"
        record = json.loads(record_path.read_text()) if record_path.is_file() else None
        if resume and job["id"] not in reset_ids and record and record.get("state") in ("generated", "accepted"):
            raw = Path(record["raw_path"])
            if not raw.is_file() or file_hash(raw) != record.get("raw_sha256"):
                raise RuntimeError(f"{job['id']}: cached raw file is missing or changed; explicitly retry this region, or use a new work directory")
            # Validate content again; file existence is never a resume criterion.
            inspect(raw, job)
            records[job["id"]] = {**record, "resumed": True}
            journal.event("resume", job=job["id"], signature=job["signature"])
        else:
            pending.append((job, folder, 0))

    effective_workers, failure = workers, None

    def execute(job, folder, attempt):
        raw = folder / f"attempt-{time.time_ns()}.png"
        begin = time.monotonic()
        row = {"schema_version": SCHEMA, "job": job["id"],
               "signature": job["signature"], "state": "running",
               "started_at": utc_now(), "attempt": attempt + 1,
               "raw_path": str(raw.resolve()), "resumed": False}
        atomic_json(folder / "record.json", row)
        journal.event("generate_start", job=job["id"], attempt=attempt + 1)
        try:
            generate(job, raw)
            result = inspect(raw, job)
            row.update(result)
            row.update(state="generated", raw_sha256=file_hash(raw), finished_at=utc_now(),
                       generation_seconds=time.monotonic() - begin)
            atomic_json(folder / "record.json", row)
            journal.event("generate_complete", job=job["id"],
                          seconds=row["generation_seconds"], native=row.get("native_dimensions"))
            return row
        except Exception as error:
            if raw.is_file():
                try:
                    from PIL import Image
                    with Image.open(raw) as opened:
                        row.update(native_dimensions=list(opened.size), native_mode=opened.mode)
                except (OSError, ValueError):
                    pass
            row.update(state="failed", error=str(error), finished_at=utc_now(),
                       generation_seconds=time.monotonic() - begin)
            atomic_json(folder / "record.json", row)
            journal.event("generate_failed", job=job["id"], error=str(error),
                          seconds=row["generation_seconds"])
            raise

    with ThreadPoolExecutor(max_workers=workers) as executor:
        active = {}
        while pending or active:
            while pending and not failure and len(active) < effective_workers:
                job, folder, attempt = pending.pop(0)
                active[executor.submit(execute, job, folder, attempt)] = (job, folder, attempt)
            if not active:
                break
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                job, folder, attempt = active.pop(future)
                try:
                    records[job["id"]] = future.result()
                except Exception as error:
                    if is_rate_limit(error) and attempt < retries and not failure:
                        effective_workers = 1
                        journal.event("concurrency_reduced", workers=1, reason="rate_limit")
                        # Drain in-flight peers before the bounded serial retry.
                        pending.append((job, folder, attempt + 1))
                        if retry_delay:
                            time.sleep(min(60.0, retry_delay * (2 ** attempt)))
                    else:
                        failure = RuntimeError(f"{job['id']}: {error}; valid peers are saved; rerun with --resume")
        # The pool drains already submitted peers; no new work after a hard failure.
    journal.event("generation_stage_end", wall_seconds=time.monotonic() - started,
                  generated=len(records), failed=bool(failure), effective_workers=effective_workers)
    if failure:
        raise failure
    return [records[job["id"]] for job in jobs]


def scoped_profile(cli, quality="high", include_auth=True):
    auth_path = Path(os.environ.get("CODEX_AUTH_FILE", str(Path.home()/".codex/auth.json")))
    auth_revision = auth_path.stat().st_mtime_ns if auth_path.is_file() else None
    return signature({"schema": SCHEMA, "cli": file_hash(cli), "model": "gpt-image-2",
                      "quality": quality,
                      "local_auth_revision": auth_revision if include_auth else None,
                      "backend": os.environ.get("CODEX_IMAGES_BASE_URL", "codex-default")})


def save_capacity(path, records, scope):
    valid = [r for r in records if r.get("native_dimensions")]
    if valid:
        atomic_json(path, {"schema_version": SCHEMA, "scope": scope, "observed_at": time.time(),
                          "observations": [{"native": r["native_dimensions"]} for r in valid],
                          "safe_pixels": int(min(r["native_dimensions"][0] * r["native_dimensions"][1]
                                                 for r in valid) * 0.8)})


def native_budget(profile=None, scope=None, default=1_200_000):
    if not profile:
        return default
    row = json.loads(Path(profile).read_text())
    if row.get("schema_version") != SCHEMA or row.get("scope") != scope:
        raise ValueError("Native profile belongs to a different backend/CLI/model/quality.")
    if not 0 <= time.time() - row.get("observed_at", 0) <= 72 * 3600:
        raise ValueError("Native profile is expired; revalidate a representative pilot.")
    return min(8_294_400, max(655_360, int(row["safe_pixels"])))


def add_runtime_arguments(parser):
    parser.add_argument("--jobs", type=int, default=2, choices=range(1, 5), help="Bounded generation workers; composition stays serial.")
    parser.add_argument("--native-profile", type=Path, help="Recent, scoped actual-return profile; never a theoretical API limit.")
    parser.add_argument("--native-pixel-budget", type=int, default=1_200_000, help="Conservative delivery crop budget until a pilot is observed.")
    parser.add_argument("--alignment", choices=("none", "translation", "affine"), default="translation")
    parser.add_argument("--max-shift", type=float, default=24)
    parser.add_argument("--max-edge-distance", type=float, default=5, help="95th percentile edge displacement limit in delivery pixels.")
    parser.add_argument("--seam-mode", choices=("feather", "minimum-error"), default="minimum-error")
    parser.add_argument("--retry-region",action="append",default=[],help="Explicitly regenerate only this region name/job ID; old native attempts remain immutable.")


def runtime_flags(args):
    flags = []
    for key in ("jobs", "native_pixel_budget", "alignment", "max_shift", "max_edge_distance", "seam_mode"):
        flags.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
    if args.native_profile:
        flags.extend(["--native-profile", str(args.native_profile)])
    for region in args.retry_region:
        if region!="global-anchor":flags.extend(["--retry-region",region])
    return flags
