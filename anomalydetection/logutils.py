import io
import logging
import time
from typing import Optional


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s" if hours else f"{minutes}m{seconds:02d}s"


class LogProgress:
    """
    Drop-in replacement for `tqdm(iterable, ...)` that emits a single log line every `interval_s` seconds
    instead of redrawing a progress bar (which is unreadable in non-interactive consoles, e.g. kubernetes logs).
    Every call is counted as one pass over the data (e.g. a training epoch), if `pass_name` is set.
    TODO: only needed as a workaround for movementpredictor using tqdm; remove once it reports progress via logging upstream
    """

    def __init__(self, logger: logging.Logger, desc: str, total: Optional[int] = None,
                 pass_name: Optional[str] = None, interval_s: float = 60) -> None:
        self._logger = logger
        self._desc = desc
        self._total = total
        self._pass_name = pass_name
        self._interval_s = interval_s
        self._passes = 0

    def __call__(self, iterable, *args, **kwargs):
        self._passes += 1
        label = f"{self._desc} {self._pass_name} {self._passes}" if self._pass_name else self._desc
        return self._iterate(iterable, label)

    def _iterate(self, iterable, label: str):
        start = last_log = time.monotonic()
        count = 0
        try:
            for item in iterable:
                yield item
                count += 1
                now = time.monotonic()
                if now - last_log >= self._interval_s:
                    last_log = now
                    self._logger.info(f"{label}: {self._progress(count, now - start)}")
        finally:
            self._logger.info(f"{label}: done after {count} batches in {_format_duration(time.monotonic() - start)}")

    def _progress(self, count: int, elapsed: float) -> str:
        rate = count / elapsed if elapsed > 0 else 0
        if not self._total:
            return f"{count} batches, {rate:.1f} batches/s, elapsed {_format_duration(elapsed)}"
        eta = (self._total - count) / rate if rate > 0 else 0
        return (f"{count}/{self._total} batches ({100 * count / self._total:.0f}%), {rate:.1f} batches/s, "
                f"elapsed {_format_duration(elapsed)}, eta {_format_duration(eta)}")


class LogWriter(io.TextIOBase):
    """File-like object that forwards every written line to a logger (e.g. to route library print() output into the log)."""

    def __init__(self, logger: logging.Logger, level: int = logging.INFO) -> None:
        self._logger = logger
        self._level = level
        self._buffer = ""

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        self._buffer += s
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            if line.strip():
                self._logger.log(self._level, line.strip())
        return len(s)

    def flush(self) -> None:
        if self._buffer.strip():
            self._logger.log(self._level, self._buffer.strip())
        self._buffer = ""
