"""FastAPI app factory.

Import-safety is a hard requirement: importing this module (or the module-level
``app``) must NOT construct an embedder/LLM or need API keys. The real service
is built LAZILY on the first request that needs it (and only then are settings
+ providers touched). Tests bypass all of that by overriding the ``get_service``
dependency with a fake.
"""

from __future__ import annotations

from fastapi import FastAPI

from regulumen.api.router import get_service, router
from regulumen.api.service import RegulumenService

_TITLE = "ReguLumen API"
_DESCRIPTION = (
    "Citation-faithful regulatory RAG: grounded Q&A with strict abstention and "
    "IEC 62304 software-safety classification. Decision support only."
)
_VERSION = "0.1.0"


def create_app(service: RegulumenService | None = None) -> FastAPI:
    """Build the FastAPI app and wire the service dependency.

    If ``service`` is provided it is used as-is (handy for tests/embedding the
    app). Otherwise the service is built LAZILY from settings on first use, so
    importing the app never needs keys or constructs providers.
    """
    app = FastAPI(title=_TITLE, description=_DESCRIPTION, version=_VERSION)
    app.include_router(router)

    if service is not None:
        app.dependency_overrides[get_service] = lambda: service
        return app

    # Lazy singleton: built on first request, cached on app state thereafter.
    def _provide_service() -> RegulumenService:
        existing = getattr(app.state, "service", None)
        if existing is None:
            from regulumen.core.config import get_settings  # noqa: PLC0415

            existing = RegulumenService.build_from_settings(get_settings())
            app.state.service = existing
        return existing

    app.dependency_overrides[get_service] = _provide_service
    return app


# Module-level app for uvicorn. Import-safe: no providers built here.
app = create_app()


__all__ = ["app", "create_app"]
