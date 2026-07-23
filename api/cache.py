# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Redis caching layer for the hot-path /wallet/{address} endpoint.

Configuration:
    - TTL: 1 hour (3600s) for demo purposes
    - Only the hot-path GET /wallet/{address} uses Redis caching
    - /explain/{address} is allowed 5–15s (no caching — GNNExplainer is slow by design)

Graceful degradation: if Redis is unavailable, all operations are no-ops.
"""

import os
import json
import logging
import redis

logger = logging.getLogger(__name__)


class RedisCache:
    """Thread-safe Redis cache wrapper with graceful fallback."""

    def __init__(self, host: str = None, port: int = None):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", 6379))
        self._client = None
        self._available = None

    @property
    def client(self):
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
        return self._client

    @property
    def is_available(self) -> bool:
        if self._available is None:
            _ = self.client  # trigger connection attempt
        return self._available

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

    def flush_all(self):
        """Flush entire cache — use with caution."""
        if not self.is_available:
            return
        try:
            self.client.flushdb()
            logger.info("Redis cache flushed.")
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
