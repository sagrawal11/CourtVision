"""Shared slowapi rate limiter.

Lives in its own module so both main.py and the api.* routers import the SAME
Limiter instance without a circular import.

Store: defaults to in-process memory, which is correct ONLY while the backend
runs a single instance (current ECS desired_count=1). Before scaling past one
task, set RATE_LIMIT_STORAGE_URI to a shared store (e.g. redis://host:6379/0)
so limits are enforced across instances.

Key: the real client IP. Behind the ALB, request.client.host is the balancer,
so we read the LAST X-Forwarded-For entry — the ALB appends the true client IP
after any client-supplied value, so a forged header cannot move the key.
"""

import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return get_remote_address(request)


limiter = Limiter(
    key_func=client_ip,
    storage_uri=os.getenv("RATE_LIMIT_STORAGE_URI", "memory://"),
)
