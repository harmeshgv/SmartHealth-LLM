class GravixLayerError(Exception):
    """Base SDK exception"""
    pass

class GravixLayerAuthenticationError(GravixLayerError):
    pass

class GravixLayerRateLimitError(GravixLayerError):
    pass

class GravixLayerServerError(GravixLayerError):
    pass

class GravixLayerBadRequestError(GravixLayerError):
    pass

class GravixLayerConnectionError(GravixLayerError):
    pass
