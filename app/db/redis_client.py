"""
Redis client configuration and connection management.
"""
import redis
from redis import ConnectionPool
from typing import Optional, Any, Dict, List
import json
from app.core.config import settings
from app.utils.logger import app_logger


class RedisClient:
    """Redis client wrapper for caching and real-time data."""
    
    def __init__(self):
        """Initialize Redis connection pool."""
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[redis.Redis] = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Redis."""
        try:
            # Parse Redis URL
            redis_url = settings.REDIS_URL
            
            # Create connection pool
            self.pool = ConnectionPool.from_url(
                redis_url,
                max_connections=settings.REDIS_POOL_SIZE,
                decode_responses=settings.REDIS_DECODE_RESPONSES,
                socket_keepalive=True,
                socket_keepalive_options={
                    1: 1,  # TCP_KEEPIDLE
                    2: 1,  # TCP_KEEPINTVL
                    3: 3,  # TCP_KEEPCNT
                }
            )
            
            # Create Redis client
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            self.client.ping()
            app_logger.info("Successfully connected to Redis")
            
        except Exception as e:
            app_logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            value = self.client.get(key)
            if value and settings.REDIS_DECODE_RESPONSES:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value
        except Exception as e:
            app_logger.error(f"Redis GET error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set value in Redis with optional expiration."""
        try:
            if not isinstance(value, str):
                value = json.dumps(value)
            return self.client.set(key, value, ex=expire)
        except Exception as e:
            app_logger.error(f"Redis SET error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            app_logger.error(f"Redis DELETE error for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            app_logger.error(f"Redis EXISTS error for key {key}: {e}")
            return False
    
    def hset(self, name: str, key: str, value: Any) -> bool:
        """Set hash field in Redis."""
        try:
            if not isinstance(value, str):
                value = json.dumps(value)
            return bool(self.client.hset(name, key, value))
        except Exception as e:
            app_logger.error(f"Redis HSET error for {name}:{key}: {e}")
            return False
    
    def hget(self, name: str, key: str) -> Optional[Any]:
        """Get hash field from Redis."""
        try:
            value = self.client.hget(name, key)
            if value:
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            return value
        except Exception as e:
            app_logger.error(f"Redis HGET error for {name}:{key}: {e}")
            return None
    
    def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all hash fields from Redis."""
        try:
            data = self.client.hgetall(name)
            result = {}
            for key, value in data.items():
                try:
                    result[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    result[key] = value
            return result
        except Exception as e:
            app_logger.error(f"Redis HGETALL error for {name}: {e}")
            return {}
    
    def lpush(self, key: str, *values: Any) -> int:
        """Push values to list."""
        try:
            encoded_values = []
            for value in values:
                if not isinstance(value, str):
                    value = json.dumps(value)
                encoded_values.append(value)
            return self.client.lpush(key, *encoded_values)
        except Exception as e:
            app_logger.error(f"Redis LPUSH error for key {key}: {e}")
            return 0
    
    def lrange(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get range of list elements."""
        try:
            values = self.client.lrange(key, start, end)
            result = []
            for value in values:
                try:
                    result.append(json.loads(value))
                except (json.JSONDecodeError, TypeError):
                    result.append(value)
            return result
        except Exception as e:
            app_logger.error(f"Redis LRANGE error for key {key}: {e}")
            return []
    
    def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel."""
        try:
            if not isinstance(message, str):
                message = json.dumps(message)
            return self.client.publish(channel, message)
        except Exception as e:
            app_logger.error(f"Redis PUBLISH error for channel {channel}: {e}")
            return 0
    
    def subscribe(self, *channels: str):
        """Subscribe to channels."""
        try:
            pubsub = self.client.pubsub()
            pubsub.subscribe(*channels)
            return pubsub
        except Exception as e:
            app_logger.error(f"Redis SUBSCRIBE error: {e}")
            return None
    
    def close(self):
        """Close Redis connection."""
        try:
            if self.client:
                self.client.close()
            if self.pool:
                self.pool.disconnect()
            app_logger.info("Redis connection closed")
        except Exception as e:
            app_logger.error(f"Error closing Redis connection: {e}")
    
    def ping(self) -> bool:
        """Test Redis connection."""
        try:
            return self.client.ping()
        except Exception:
            return False


# Create global Redis client instance
redis_client = RedisClient()


def get_redis() -> RedisClient:
    """Get Redis client instance."""
    return redis_client