# api/deps.py
"""
Dependency injection for Neo4j driver and Redis client.
Used as FastAPI Depends() in all routers.

Compliance Disclaimer: This system is a research and portfolio
demonstration only. Not a certified AML/CFT compliance tool.
"""
from neo4j import AsyncGraphDatabase
from redis.asyncio import Redis
import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


@lru_cache
def get_neo4j_driver():
    return AsyncGraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "changeme_in_prod"),
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
