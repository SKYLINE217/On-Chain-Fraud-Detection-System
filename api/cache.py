# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Redis caching layer for the hot-path /wallet/{address} endpoint.

Configuration:
    - TTL: 1 hour (3600s) for demo purposes
    - Only the hot-path GET /wallet/{address} uses Redis caching
    - /explain/{address} is allowed 5–15s (no caching — GNNExplainer is slow by design)

Graceful degradation: if Redis is unavailable, all operations are no-ops.

Fixes applied:
  BUG-08: flush_scored_keys() added for targeted key deletion (avoids full flushdb)
  BUG-28: Exponential backoff — does not retry connection on every request after failure
"""

import os
import json
import logging
import time
import redis

logger = logging.getLogger(__name__)


class RedisCache:
    """Thread-safe Redis cache wrapper with graceful fallback."""

    # BUG-28: Retry interval — wait at least this many seconds before attempting reconnect
    _RETRY_INTERVAL = 30  # seconds

    def __init__(self, host: str = None, port: int = None):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", 6379))
        self._client = None
        self._available = None
        self._last_retry: float = 0  # BUG-28: timestamp of last failed connection attempt

    @property
    def client(self):
        # BUG-28: If previously failed, don't retry until _RETRY_INTERVAL has elapsed
        if self._available is False:
            if time.time() - self._last_retry < self._RETRY_INTERVAL:
                return None  # Skip retry — still in backoff window

        if self._client is None:
            try:
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                self._client.ping()
                self._available = True
                logger.info("Redis connected: %s:%d", self.host, self.port)
            except (redis.ConnectionError, redis.TimeoutError) as e:
                logger.warning("Redis unavailable (%s) — caching disabled.", e)
                self._client = None
                self._available = False
                self._last_retry = time.time()  # BUG-28: record failure timestamp
        return self._client

    @property
    def is_available(self) -> bool:
        if self._available is None:
            _ = self.client  # trigger connection attempt
        return bool(self._available)

    def get(self, key: str) -> dict | None:
        """
        Retrieve cached value. Returns None on miss or if Redis is down.
        """
        if not self.is_available:
            return None
        try:
            val = self.client.get(key)
            if val:
                return json.loads(val)
            return None
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.warning("Redis GET failed for key=%s: %s", key, e)
            return None

    def set(self, key: str, value: dict, ttl: int = 3600):
        """
        Store value in cache with TTL (default 1 hour).
        No-op if Redis is unavailable.
        """
        if not self.is_available:
            return
        try:
            self.client.setex(key, ttl, json.dumps(value, default=str))
        except redis.RedisError as e:
            logger.warning("Redis SET failed for key=%s: %s", key, e)

    def delete(self, key: str):
        """Delete a cached key."""
        if not self.is_available:
            return
        try:
            self.client.delete(key)
        except redis.RedisError as e:
            logger.warning("Redis DELETE failed for key=%s: %s", key, e)

    def flush_scored_keys(self, txid_list: list):
        """
        BUG-08 Fix: Delete ONLY the score cache keys for re-scored nodes.
        Avoids the nuclear flushdb() which would wipe unrelated Redis data.

        Args:
            txid_list: List of txIds whose cache entries should be invalidated.
        """
        if not self.is_available:
            return
        try:
            keys = [f"score:{txid}" for txid in txid_list]
            if keys:
                self.client.delete(*keys)
            logger.info("Flushed %d score cache keys", len(keys))
        except redis.RedisError as e:
            logger.warning("Redis targeted flush failed: %s", e)

    def flush_all(self):
        """
        Flush entire cache database.
        WARNING (BUG-08): This nukes ALL keys in the Redis DB, including
        any non-score data. Prefer flush_scored_keys() for post-batch cleanup.
        Use this only for full cache resets.
        """
        if not self.is_available:
            return
        try:
            self.client.flushdb()
            logger.info("Redis cache flushed (ALL keys).")
        except redis.RedisError as e:
            logger.warning("Redis FLUSHDB failed: %s", e)

    def get_stats(self) -> dict:
        """Return cache hit/miss statistics from Redis INFO."""
        if not self.is_available:
            return {"status": "unavailable"}
        try:
            info = self.client.info("stats")
            return {
                "status": "connected",
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0)
                    / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
                ),
            }
        except redis.RedisError:
            return {"status": "error"}


# Module-level singleton for use in health checks
cache = RedisCache()
