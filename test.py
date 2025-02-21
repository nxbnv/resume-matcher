import redis
import os
from dotenv import load_dotenv

load_dotenv()
REDIS_URL = os.getenv("REDIS_URL", "redis://default:J16bosr7jhwQQxj4FwQjJkQFFr3CMWCb@redis-19698.c311.eu-central-1-1.ec2.redns.redis-cloud.com:19698/0")
client = redis.Redis.from_url(REDIS_URL)

try:
    client.ping()
    print("✅ Redis is working!")
except Exception as e:
    print(f"❌ Redis error: {e}")
