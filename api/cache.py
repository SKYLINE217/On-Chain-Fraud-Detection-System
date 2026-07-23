# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

import os
import json
import redis

class RedisCache:
    def __init__(self, host: str = None, port: int = None):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", 6379))
        try:
            self.client = redis.Redis(host=self.host, port=self.port, decode_responses=True)
        except Exception:
            self.client = None

    def get(self, key: str):
        if not self.client:
            return None
        try:
            val = self.client.get(key)
            return json.loads(val) if val else None
        except Exception:
            return None

    def set(self, key: str, value: dict, ttl: int = 3600):
        if not self.client:
            return
        try:
            self.client.setex(key, ttl, json.dumps(value))
        except Exception:
            pass
