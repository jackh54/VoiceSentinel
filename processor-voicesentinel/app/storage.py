from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_storage_backends: dict[str, StorageBackend] = {}


def r2_storage_configured() -> bool:
    return bool(os.environ.get("VOICESENTINEL_R2_BUCKET", "").strip())


class StorageBackend(ABC):
    @abstractmethod
    def write_bytes(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_bytes(self, key: str) -> Optional[bytes]:
        raise NotImplementedError

    @abstractmethod
    def append_bytes(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    def write_text(self, key: str, text: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_text(self, key: str) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def read_text_lines(self, key: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def list_keys(self, prefix: str, suffix: str = "") -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_mtime(self, key: str) -> Optional[float]:
        raise NotImplementedError

    @abstractmethod
    def ensure_prefix(self, prefix: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_dir(self, prefix: str) -> bool:
        raise NotImplementedError


def _normalize_key(key: str) -> str:
    return key.replace("\\", "/").lstrip("/")


class LocalFilesystemStore(StorageBackend):
    def __init__(self, root: str):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        rel = _normalize_key(key)
        path = self.root / rel
        if self.root.resolve() not in path.resolve().parents and path.resolve() != self.root.resolve():
            raise ValueError(f"Invalid storage key: {key}")
        return path

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def read_bytes(self, key: str) -> Optional[bytes]:
        path = self._path(key)
        if not path.is_file():
            return None
        return path.read_bytes()

    def append_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "ab") as f:
            f.write(data)

    def write_text(self, key: str, text: str) -> None:
        self.write_bytes(key, text.encode("utf-8"))

    def read_text(self, key: str) -> Optional[str]:
        raw = self.read_bytes(key)
        if raw is None:
            return None
        return raw.decode("utf-8")

    def read_text_lines(self, key: str) -> List[str]:
        text = self.read_text(key)
        if text is None:
            return []
        return text.splitlines()

    def list_keys(self, prefix: str, suffix: str = "") -> List[str]:
        base = self._path(prefix)
        if not base.exists():
            return []
        if base.is_file():
            rel = _normalize_key(str(base.relative_to(self.root)))
            return [rel] if (not suffix or rel.endswith(suffix)) else []
        keys: List[str] = []
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel = _normalize_key(str(path.relative_to(self.root)))
            if suffix and not rel.endswith(suffix):
                continue
            keys.append(rel)
        return sorted(keys)

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.is_file():
            path.unlink(missing_ok=True)

    def get_mtime(self, key: str) -> Optional[float]:
        path = self._path(key)
        if not path.is_file():
            return None
        return path.stat().st_mtime

    def ensure_prefix(self, prefix: str) -> None:
        self._path(prefix).mkdir(parents=True, exist_ok=True)

    def is_dir(self, prefix: str) -> bool:
        return self._path(prefix).is_dir()


class R2Store(StorageBackend):
    def __init__(self, bucket: str, endpoint_url: str, access_key: str, secret_key: str):
        self.bucket = bucket
        self.endpoint_url = endpoint_url.rstrip("/")
        self.access_key = access_key
        self.secret_key = secret_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name="auto",
            )
        return self._client

    def _object_key(self, key: str) -> str:
        return _normalize_key(key)

    def write_bytes(self, key: str, data: bytes) -> None:
        self._get_client().put_object(Bucket=self.bucket, Key=self._object_key(key), Body=data)

    def read_bytes(self, key: str) -> Optional[bytes]:
        try:
            response = self._get_client().get_object(Bucket=self.bucket, Key=self._object_key(key))
            return response["Body"].read()
        except Exception as exc:
            from botocore.exceptions import ClientError

            if isinstance(exc, ClientError):
                code = exc.response.get("Error", {}).get("Code")
                if code in ("NoSuchKey", "404", "NotFound"):
                    return None
            raise

    def append_bytes(self, key: str, data: bytes) -> None:
        existing = self.read_bytes(key) or b""
        self.write_bytes(key, existing + data)

    def write_text(self, key: str, text: str) -> None:
        self.write_bytes(key, text.encode("utf-8"))

    def read_text(self, key: str) -> Optional[str]:
        raw = self.read_bytes(key)
        if raw is None:
            return None
        return raw.decode("utf-8")

    def read_text_lines(self, key: str) -> List[str]:
        text = self.read_text(key)
        if text is None:
            return []
        return text.splitlines()

    def list_keys(self, prefix: str, suffix: str = "") -> List[str]:
        client = self._get_client()
        normalized = self._object_key(prefix)
        if normalized and not normalized.endswith("/"):
            normalized += "/"
        keys: List[str] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=normalized):
            for item in page.get("Contents") or []:
                object_key = item["Key"]
                if suffix and not object_key.endswith(suffix):
                    continue
                keys.append(object_key)
        return sorted(keys)

    def delete(self, key: str) -> None:
        self._get_client().delete_object(Bucket=self.bucket, Key=self._object_key(key))

    def get_mtime(self, key: str) -> Optional[float]:
        try:
            response = self._get_client().head_object(Bucket=self.bucket, Key=self._object_key(key))
            return response["LastModified"].timestamp()
        except Exception as exc:
            from botocore.exceptions import ClientError

            if isinstance(exc, ClientError):
                code = exc.response.get("Error", {}).get("Code")
                if code in ("NoSuchKey", "404", "NotFound"):
                    return None
            raise

    def ensure_prefix(self, prefix: str) -> None:
        return

    def is_dir(self, prefix: str) -> bool:
        normalized = self._object_key(prefix)
        if not normalized.endswith("/"):
            normalized += "/"
        response = self._get_client().list_objects_v2(
            Bucket=self.bucket, Prefix=normalized, MaxKeys=1
        )
        return bool(response.get("KeyCount"))


def create_storage_backend(root: str) -> StorageBackend:
    bucket = os.environ.get("VOICESENTINEL_R2_BUCKET", "").strip()
    if bucket:
        endpoint = os.environ.get("R2_ENDPOINT_URL", "").strip()
        access_key = os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
        if not endpoint or not access_key or not secret_key:
            raise RuntimeError(
                "VOICESENTINEL_R2_BUCKET is set but R2_ENDPOINT_URL, "
                "AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY are required"
            )
        logger.info("Storage backend: R2 (bucket=%s)", bucket)
        return R2Store(bucket, endpoint, access_key, secret_key)
    logger.info("Storage backend: local filesystem (root=%s)", root)
    return LocalFilesystemStore(root)


def init_storage(root: str) -> StorageBackend:
    global _storage_backend
    _storage_backend = create_storage_backend(root)
    return _storage_backend


def get_storage() -> StorageBackend:
    if _storage_backend is None:
        raise RuntimeError("Storage backend not initialized")
    return _storage_backend


def storage_key(*parts: str) -> str:
    cleaned = [_normalize_key(part).strip("/") for part in parts if part]
    return "/".join(cleaned)
