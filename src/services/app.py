"""
Aplicación principal FastAPI para Aula AI Tutor.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from src.services.routers import solver, tutor
from src.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación."""
    logger.info("Iniciando servicios de Aula AI Tutor...")
    yield
    logger.info("Apagando servicios...")


def create_app() -> FastAPI:
    """Crea y configura la aplicación FastAPI."""
    app = FastAPI(
        title="Aula AI Tutor API",
        description="API para sistema de tutoría socrática con agentes",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # En producción restringir esto
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Incluir routers
    app.include_router(solver.router, prefix="/solver", tags=["Solver"])
    app.include_router(tutor.router, prefix="/tutor", tags=["Tutor"])
    
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "version": "0.1.0"}
        
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.services.app:app",
        host=settings.a2a.solver_host,
        port=settings.a2a.solver_port,
        reload=settings.debug,
    )
