"""The one model-kb write lock: an exclusive flock on <root>/.ingest.lock.

pipeline.py (both crons) and recipe_store.put take this same lock, so a recipe
write never interleaves with a generation rebuild.
"""
from __future__ import annotations

import fcntl
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

LOCK_NAME = ".ingest.lock"
EXIT_LOCKED = 75  # EX_TEMPFAIL — "busy, retry"


def acquire_lock(root: Path, timeout_s: float, *, who: str = "") -> IO[str] | str:
    """Hold the lock (returns the open file; close it to release) or, after
    timeout_s, return the holder's "<pid> <since> [who]" text instead."""
    fh = open(Path(root) / LOCK_NAME, "a+")
    deadline = time.monotonic() + max(timeout_s, 0.0)
    announced = False
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fh.seek(0)
            fh.truncate()
            stamp = datetime.now(timezone.utc).isoformat()
            fh.write(f"{os.getpid()} {stamp}{' ' + who if who else ''}\n")
            fh.flush()
            return fh
        except BlockingIOError:
            if time.monotonic() >= deadline:
                fh.seek(0)
                holder = fh.read().strip() or "?"
                fh.close()
                return holder
            if not announced:
                fh.seek(0)
                print(f"waiting for ingest lock (held by {fh.read().strip() or '?'})", flush=True)
                announced = True
            time.sleep(min(5.0, max(deadline - time.monotonic(), 0.05)))
