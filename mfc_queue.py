from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import redis

DEFAULT_NAMESPACE = "mfc_clappia"
DEFAULT_DEDUPE_TTL_SECONDS = 1800


def _normalize_namespace(value: str | None) -> str:
    cleaned = (value or DEFAULT_NAMESPACE).strip().strip(":")
    return cleaned or DEFAULT_NAMESPACE


@dataclass(frozen=True)
class MFCQueueConfig:
    redis_url: str
    queue_name: str = "mfc_clappia_jobs"
    failed_queue_name: str = "mfc_clappia_failed"
    namespace: str = DEFAULT_NAMESPACE
    jobs_key: str | None = None
    processing_key: str | None = None
    failed_key: str | None = None
    dedupe_prefix: str | None = None
    dedupe_ttl_seconds: int = DEFAULT_DEDUPE_TTL_SECONDS

    def __post_init__(self) -> None:
        namespace = _normalize_namespace(self.namespace)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "jobs_key", self.jobs_key or f"{namespace}:jobs")
        object.__setattr__(self, "processing_key", self.processing_key or f"{namespace}:processing")
        object.__setattr__(self, "failed_key", self.failed_key or f"{namespace}:failed")
        object.__setattr__(self, "dedupe_prefix", self.dedupe_prefix or f"{namespace}:dedupe:")


class MFCQueueClient:
    def __init__(self, config: MFCQueueConfig):
        self.config = config
        self._client: redis.Redis | None = None

    def is_configured(self) -> bool:
        return bool(self.config.redis_url)

    def describe(self) -> dict[str, Any]:
        return {
            "configured": self.is_configured(),
            "queue_name": self.config.queue_name,
            "failed_queue_name": self.config.failed_queue_name,
            "namespace": self.config.namespace,
            "jobs_key": self.config.jobs_key,
            "processing_key": self.config.processing_key,
            "failed_key": self.config.failed_key,
            "dedupe_prefix": self.config.dedupe_prefix,
            "dedupe_ttl_seconds": self.config.dedupe_ttl_seconds,
        }

    def connect(self) -> redis.Redis:
        if not self.is_configured():
            raise RuntimeError("REDIS_URL is not configured.")
        if self._client is None:
            self._client = redis.Redis.from_url(
                self.config.redis_url,
                decode_responses=True,
            )
        return self._client

    def ping(self) -> bool:
        return bool(self.connect().ping())

    def _dedupe_hash(self, job_payload: dict[str, Any]) -> str:
        signature_payload = {
            "submission_id": job_payload.get("submission_id"),
            "workplace_id": job_payload.get("workplace_id"),
            "targets": job_payload.get("targets", {}),
        }
        serialized = json.dumps(
            signature_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _dedupe_key(self, dedupe_hash: str) -> str:
        return f"{self.config.dedupe_prefix}{dedupe_hash}"

    def enqueue(self, job_payload: dict[str, Any]) -> dict[str, Any]:
        client = self.connect()
        job = dict(job_payload)
        dedupe_hash = str(job.get("dedupe_hash") or self._dedupe_hash(job))
        job["dedupe_hash"] = dedupe_hash
        raw_job = json.dumps(job, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        dedupe_key = self._dedupe_key(dedupe_hash)
        stored = client.set(
            dedupe_key,
            job.get("job_id") or dedupe_hash,
            nx=True,
            ex=self.config.dedupe_ttl_seconds,
        )
        if not stored:
            return {
                "enqueued": False,
                "duplicate": True,
                "job_id": client.get(dedupe_key) or job.get("job_id"),
                "dedupe_hash": dedupe_hash,
                "queue_name": self.config.queue_name,
                "jobs_key": self.config.jobs_key,
            }

        client.lpush(self.config.jobs_key, raw_job)
        return {
            "enqueued": True,
            "duplicate": False,
            "job_id": job.get("job_id"),
            "dedupe_hash": dedupe_hash,
            "queue_name": self.config.queue_name,
            "jobs_key": self.config.jobs_key,
        }

    def dequeue(self, timeout_seconds: int = 5) -> dict[str, Any] | None:
        raw_job = self.connect().brpoplpush(
            self.config.jobs_key,
            self.config.processing_key,
            timeout_seconds,
        )
        if raw_job is None:
            return None
        return {
            "raw_job": raw_job,
            "job": json.loads(raw_job),
        }

    def complete(self, *, raw_job: str, job_payload: dict[str, Any]) -> None:
        client = self.connect()
        client.lrem(self.config.processing_key, 1, raw_job)
        dedupe_hash = job_payload.get("dedupe_hash")
        if dedupe_hash:
            client.delete(self._dedupe_key(str(dedupe_hash)))

    def fail(
        self,
        *,
        raw_job: str,
        job_payload: dict[str, Any],
        failure_payload: dict[str, Any],
    ) -> dict[str, Any]:
        client = self.connect()
        client.lrem(self.config.processing_key, 1, raw_job)
        dedupe_hash = job_payload.get("dedupe_hash")
        if dedupe_hash:
            client.delete(self._dedupe_key(str(dedupe_hash)))

        record = {
            "failed_at": int(time.time()),
            "job": job_payload,
            "failure": failure_payload,
        }
        client.lpush(
            self.config.failed_key,
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        )
        return record
