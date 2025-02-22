from celery import Celery
import os

# ✅ Use Redis as both broker and result backend
REDIS_URL = os.getenv("REDIS_URL", "redis://default:J16bosr7jhwQQxj4FwQjJkQFFr3CMWCb@redis-19698.c311.eu-central-1-1.ec2.redns.redis-cloud.com:19698/0")

celery = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)

# ✅ Configure Celery to persist task results
celery.conf.update(
    broker_connection_retry_on_startup=True,
    result_backend=REDIS_URL,  # ✅ This fixes 'DisabledBackend' error
    task_track_started=True,
    result_extended=True,
    result_expires=3600
)

celery.autodiscover_tasks(["tasks"])

