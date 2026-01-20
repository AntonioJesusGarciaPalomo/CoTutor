"""
Router para funcionalidades del Solver Agent.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.agents.solver import SolverAgent
from config.settings import get_settings
from src.core.types import StructuredSolution

router = APIRouter()
settings = get_settings()


class SolveRequest(BaseModel):
    problem_text: str
    domain_hint: str | None = None
    additional_context: str = ""


async def get_solver_agent():
    """Dependencia para obtener una instancia del Solver."""
    # En producción esto podría ser un singleton manejado de otra forma
    return await SolverAgent.create(settings.model_defaults.solver_model)


@router.post("/solve", response_model=StructuredSolution)
async def solve_problem(
    request: SolveRequest,
    solver: Annotated[SolverAgent, Depends(get_solver_agent)],
):
    """
    Resuelve un problema y devuelve la solución estructurada.
    """
    try:
        solution = await solver.solve(
            problem_text=request.problem_text,
            domain_hint=request.domain_hint,
            additional_context=request.additional_context,
        )
        return solution
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
