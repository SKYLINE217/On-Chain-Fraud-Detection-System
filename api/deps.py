from neo4j import AsyncGraphDatabase
from redis.asyncio import Redis
import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

@lru_cache
def get_neo4j_driver():
    neo4j_pwd = os.environ.get("NEO4J_PASSWORD")
    if not neo4j_pwd:
        raise ValueError("NEO4J_PASSWORD environment variable is not set")
    return AsyncGraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.environ.get("NEO4J_USER", "neo4j"),
            neo4j_pwd,
        ),
        max_connection_lifetime=300,
        max_connection_pool_size=50,
    )

@lru_cache
def get_redis() -> Redis:
    return Redis.from_url(
        os.environ.get("REDIS_URL", "redis://localhost:6379"),
        decode_responses=True,
    )
