"""
Router para funcionalidades del Tutor Agent.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.agents.tutor import TutorAgent
from config.settings import get_settings
from src.core.types import StructuredSolution, TutorResponse

router = APIRouter()
settings = get_settings()

# Instancia global del Tutor (para mantener estado en memoria simple)
# En producción usar Redis o DB
_tutor_agent: TutorAgent | None = None


class SessionRequest(BaseModel):
    problem_text: str
    solution: StructuredSolution


class SessionResponse(BaseModel):
    session_id: str
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


async def get_tutor_agent() -> TutorAgent:
    """Obtiene o crea el agente Tutor singleton."""
    global _tutor_agent
    if _tutor_agent is None:
        _tutor_agent = await TutorAgent.create(settings.model_defaults.tutor_model)
    return _tutor_agent


@router.post("/session", response_model=SessionResponse)
async def start_session(
    request: SessionRequest,
    tutor: TutorAgent = Depends(get_tutor_agent),
):
    """Inicia una nueva sesión de tutoría."""
    try:
        session = await tutor.start_session(
            problem_text=request.problem_text,
            solution=request.solution,
        )
        return SessionResponse(
            session_id=str(session.session_id),
            message="Sesión iniciada correctamente",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=TutorResponse)
async def chat(
    request: ChatRequest,
    tutor: TutorAgent = Depends(get_tutor_agent),
):
    """Envía un mensaje al tutor."""
    try:
        session_id = UUID(request.session_id)
        response = await tutor.respond(session_id, request.message)
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
