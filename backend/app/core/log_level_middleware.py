import logging
from fastapi import Request

class PerRouteLogLevelMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        root_logger = logging.getLogger()
        original_level = root_logger.level

        path = request.url.path

        # 🔥 RULES
        if path.startswith("/debug"):
            root_logger.setLevel(logging.DEBUG)
        else:
            root_logger.setLevel(logging.ERROR)

        try:
            await self.app(scope, receive, send)
        finally:
            root_logger.setLevel(original_level)
