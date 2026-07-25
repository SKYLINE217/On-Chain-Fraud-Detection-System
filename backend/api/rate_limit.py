import logging

logger = logging.getLogger(__name__)

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    limiter = Limiter(key_func=get_remote_address)
    _rate_limiting_available = True
    logger.info("slowapi rate limiting enabled")
except ImportError:
    class DummyLimiter:
        def limit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    limiter = DummyLimiter()
    RateLimitExceeded = Exception
    _rate_limiting_available = False
    logger.warning("slowapi not installed — rate limiting disabled.")
