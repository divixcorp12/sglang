"""Verified persistent storage for groups of file-backed tensors."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch

from sglang.srt.model_loader.weight_utils import get_lock

_FORMAT_VERSION = 1
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileTensorSpec:
    """Shape, stride, and dtype of one file-backed tensor."""

    tag: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        object.__setattr__(self, "tag", str(self.tag))
        object.__setattr__(self, "shape", tuple(int(value) for value in self.shape))
        object.__setattr__(self, "stride", tuple(int(value) for value in self.stride))
        if not self.tag:
            raise ValueError("file tensor tag must not be empty")
        if len(self.shape) != len(self.stride):
            raise ValueError("file tensor shape and stride must have the same rank")
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("file tensor dimensions must be non-negative")
        if any(value < 0 for value in self.stride):
            raise ValueError("file tensor strides must be non-negative")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("file tensor dtype must be a torch.dtype")

    @property
    def nbytes(self) -> int:
        """Storage bytes required by an offset-zero strided view."""
        if any(dimension == 0 for dimension in self.shape):
            return 0
        storage_elements = 1 + sum(
            (dimension - 1) * stride
            for dimension, stride in zip(self.shape, self.stride)
        )
        return storage_elements * torch.empty((), dtype=self.dtype).element_size()


class FileTensorCacheGroup:
    """One atomically verified group of file-backed tensor mappings."""

    def __init__(
        self,
        *,
        cache_hit: bool,
        directory: str,
        manifest_path: str,
        manifest: dict[str, Any],
        specs: tuple[FileTensorSpec, ...],
        paths: dict[str, str],
        tensors: dict[str, torch.Tensor],
        lock: Any,
    ) -> None:
        self.cache_hit = bool(cache_hit)
        self.directory = directory
        self.manifest_path = manifest_path
        self.specs = specs
        self.paths = paths
        self.tensors = tensors
        self._manifest = manifest
        self._lock = lock
        self._closed = False

    @classmethod
    def open(
        cls,
        directory: str | os.PathLike[str],
        namespace: str,
        cache_identity: Mapping[str, Any],
        specs: Sequence[FileTensorSpec],
    ) -> "FileTensorCacheGroup":
        """Open a verified hit or create a fresh all-member cache miss."""
        directory_path = os.path.realpath(os.path.expanduser(os.fspath(directory)))
        os.makedirs(directory_path, exist_ok=True)
        namespace = str(namespace)
        normalized_specs = tuple(specs)
        if not namespace:
            raise ValueError("file tensor cache namespace must not be empty")
        if not normalized_specs:
            raise ValueError("file tensor cache group must contain at least one tensor")
        tags = [spec.tag for spec in normalized_specs]
        if len(set(tags)) != len(tags):
            raise ValueError("file tensor cache tags must be unique within a group")

        manifest, digest = _build_manifest(namespace, cache_identity, normalized_specs)
        stem = f"file_tensor_cache_{_safe_component(namespace)}_{digest}"
        manifest_path = os.path.join(directory_path, f"{stem}.manifest.json")
        paths = {
            spec.tag: os.path.join(
                directory_path,
                f"{stem}_{index:03d}_{_safe_component(spec.tag)}.bin",
            )
            for index, spec in enumerate(normalized_specs)
        }
        lock_digest = hashlib.sha256(
            os.path.realpath(manifest_path).encode("utf-8")
        ).hexdigest()
        lock = get_lock(f"file-tensor-cache-{lock_digest}")
        lock.acquire()
        try:
            cache_hit = _cache_is_valid(
                manifest_path, manifest, normalized_specs, paths
            )
            if not cache_hit:
                _log_cache_event(manifest_path, manifest, "building_miss")
                _invalidate_manifest(manifest_path)
                for spec in normalized_specs:
                    _replace_sparse_file(paths[spec.tag], spec.nbytes)
            tensors = {
                spec.tag: _map_tensor(paths[spec.tag], spec)
                for spec in normalized_specs
            }
            group = cls(
                cache_hit=cache_hit,
                directory=directory_path,
                manifest_path=manifest_path,
                manifest=manifest,
                specs=normalized_specs,
                paths=paths,
                tensors=tensors,
                lock=lock,
            )
            if cache_hit:
                _log_cache_event(manifest_path, manifest, "verified_hit")
            return group
        except Exception as error:
            lock.release()
            _log_cache_event(
                manifest_path,
                manifest,
                "open_failed",
                level=logging.WARNING,
                reason=type(error).__name__,
            )
            raise

    def complete(self) -> None:
        """Durably publish every member as one complete cache group."""
        if self._closed:
            return
        temporary_path: Optional[str] = None
        try:
            for path in self.paths.values():
                _fsync_file(path)
            temporary_fd, temporary_path = tempfile.mkstemp(
                prefix=f"{os.path.basename(self.manifest_path)}.tmp",
                dir=self.directory,
            )
            try:
                with os.fdopen(temporary_fd, "w", encoding="utf-8") as stream:
                    json.dump(self._manifest, stream, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, self.manifest_path)
                temporary_path = None
                _fsync_directory(self.directory)
            finally:
                if temporary_path is not None:
                    _unlink_if_present(temporary_path)
            _log_cache_event(
                self.manifest_path, self._manifest, "published_completed_cache"
            )
        except Exception as error:
            _log_cache_event(
                self.manifest_path,
                self._manifest,
                "publication_failed",
                level=logging.WARNING,
                reason=type(error).__name__,
            )
            raise
        finally:
            self.close()

    def abort(self) -> None:
        """Invalidate this group and release its cache lock."""
        if self._closed:
            return
        try:
            _invalidate_manifest(self.manifest_path)
            _log_cache_event(self.manifest_path, self._manifest, "aborted_invalidated")
        except Exception as error:
            _log_cache_event(
                self.manifest_path,
                self._manifest,
                "invalidation_failed",
                level=logging.WARNING,
                reason=type(error).__name__,
            )
            raise
        finally:
            self.close()

    def close(self) -> None:
        """Release the group lock once without publishing a cache miss."""
        if self._closed:
            return
        self._closed = True
        lock, self._lock = self._lock, None
        if lock is not None:
            lock.release()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _build_manifest(
    namespace: str,
    cache_identity: Mapping[str, Any],
    specs: tuple[FileTensorSpec, ...],
) -> tuple[dict[str, Any], str]:
    payload = {
        "format_version": _FORMAT_VERSION,
        "namespace": namespace,
        "cache_identity": dict(cache_identity),
        "tensors": [
            {
                "tag": spec.tag,
                "shape": list(spec.shape),
                "stride": list(spec.stride),
                "dtype": str(spec.dtype),
                "nbytes": spec.nbytes,
            }
            for spec in specs
        ],
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return json.loads(canonical), hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cache_is_valid(
    manifest_path: str,
    expected_manifest: dict[str, Any],
    specs: tuple[FileTensorSpec, ...],
    paths: dict[str, str],
) -> bool:
    reason = "manifest_unreadable"
    try:
        with open(manifest_path, encoding="utf-8") as stream:
            actual_manifest = json.load(stream)
        if not _exact_json_equal(actual_manifest, expected_manifest):
            _log_cache_event(
                manifest_path,
                expected_manifest,
                "validation_miss",
                level=logging.DEBUG,
                reason="manifest_mismatch",
            )
            return False
        reason = "member_unavailable"
        for index, spec in enumerate(specs):
            actual_bytes = os.path.getsize(paths[spec.tag])
            if actual_bytes != spec.nbytes:
                _log_cache_event(
                    manifest_path,
                    expected_manifest,
                    "validation_miss",
                    level=logging.DEBUG,
                    reason=f"member_size_mismatch member={index} expected_bytes={spec.nbytes} actual_bytes={actual_bytes}",
                )
                return False
        return True
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        if reason == "manifest_unreadable" and isinstance(error, FileNotFoundError):
            reason = "manifest_missing"
        _log_cache_event(
            manifest_path,
            expected_manifest,
            "validation_miss",
            level=logging.DEBUG,
            reason=f"{reason} error={type(error).__name__}",
        )
        return False


def _log_cache_event(
    manifest_path: str,
    manifest: Mapping[str, Any],
    outcome: str,
    *,
    level: int = logging.INFO,
    reason: str | None = None,
) -> None:
    """Describe the group using its digest, without disclosing checkpoint paths."""
    if not logger.isEnabledFor(level):
        return
    key = os.path.basename(manifest_path).rsplit("_", 1)[-1][:12]
    members = manifest["tensors"]
    logger.log(
        level,
        "File tensor cache namespace=%s key=%s members=%d bytes=%d outcome=%s%s",
        _safe_component(manifest["namespace"]),
        key,
        len(members),
        sum(member["nbytes"] for member in members),
        outcome,
        "" if reason is None else f" reason={reason}",
    )


def _exact_json_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _exact_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_json_equal(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        )
    return bool(actual == expected)


def _map_tensor(path: str, spec: FileTensorSpec) -> torch.Tensor:
    storage = torch.from_file(path, shared=True, size=spec.nbytes, dtype=torch.uint8)
    return torch.as_strided(
        storage.view(spec.dtype), size=spec.shape, stride=spec.stride
    )


def _replace_sparse_file(path: str, nbytes: int) -> None:
    temporary_fd, temporary_path = tempfile.mkstemp(
        prefix=f"{os.path.basename(path)}.tmp", dir=os.path.dirname(path)
    )
    try:
        try:
            os.ftruncate(temporary_fd, nbytes)
        finally:
            os.close(temporary_fd)
        os.replace(temporary_path, path)
        temporary_path = ""
    finally:
        if temporary_path:
            _unlink_if_present(temporary_path)


def _invalidate_manifest(manifest_path: str) -> None:
    try:
        os.unlink(manifest_path)
    except FileNotFoundError:
        return
    _fsync_directory(os.path.dirname(manifest_path))


def _fsync_file(path: str) -> None:
    descriptor = os.open(path, os.O_RDWR)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(directory: str) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_if_present(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return safe[:64] or "cache"
