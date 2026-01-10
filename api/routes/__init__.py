"""API route modules."""

from api.routes.upload import router as upload_router
from api.routes.analytics import router as analytics_router
from api.routes.predictions import router as predictions_router
from api.routes.compensation import router as compensation_router
from api.routes.succession import router as succession_router
from api.routes.team import router as team_router
from api.routes.fairness import router as fairness_router
from api.routes.search import router as search_router
from api.routes.advisor import router as advisor_router

__all__ = [
    "upload_router",
    "analytics_router",
    "predictions_router",
    "compensation_router",
    "succession_router",
    "team_router",
    "fairness_router",
    "search_router",
    "advisor_router",
]
