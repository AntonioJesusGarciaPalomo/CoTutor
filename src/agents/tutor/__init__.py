"""
Agente Tutor para Aula AI Tutor.

Este módulo exporta el agente tutor y sus componentes principales.
El TutorAgent guía a estudiantes usando el método socrático,
integrándose con el sistema de guardrails para asegurar que
nunca se revelen soluciones directamente.

Ejemplo de uso:
    ```python
    import asyncio
    from src.agents.solver import SolverAgent
    from src.agents.tutor import TutorAgent

    async def main():
        # Crear agentes
        solver = await SolverAgent.create()
        tutor = await TutorAgent.create()

        # Resolver problema
        problem = "Resuelve: 2x + 3 = 7"
        solution = await solver.solve(problem)

        # Iniciar tutoría
        session = await tutor.start_session(problem, solution)

        # Interactuar con el estudiante
        response = await tutor.respond(
            session.session_id,
            "No sé cómo empezar"
        )
        print(response.content)
        # "¿Qué operación crees que necesitas para aislar la variable?"

    asyncio.run(main())
    ```
"""

from src.agents.tutor.agent import TutorAgent
from src.agents.tutor.prompts import (
    CLARIFICATION_PROMPT,
    ENCOURAGEMENT_PROMPT,
    HINT_GIVING_PROMPT,
    REDIRECTION_PROMPT,
    SOCRATIC_PROMPT,
    TUTOR_SYSTEM_PROMPT,
    VERIFICATION_PROMPT,
    format_hint_request,
    format_student_message,
    format_tutor_context,
    format_verification_request,
    get_tutor_prompt,
)
from src.agents.tutor.session_manager import SessionManager, session_manager
from src.agents.tutor.strategies import (
    StrategySelector,
    TutoringStrategy,
    strategy_selector,
)

__all__ = [
    # Agente principal
    "TutorAgent",
    # Estrategias
    "TutoringStrategy",
    "StrategySelector",
    "strategy_selector",
    # Gestor de sesiones
    "SessionManager",
    "session_manager",
    # Prompts
    "TUTOR_SYSTEM_PROMPT",
    "SOCRATIC_PROMPT",
    "HINT_GIVING_PROMPT",
    "CLARIFICATION_PROMPT",
    "ENCOURAGEMENT_PROMPT",
    "VERIFICATION_PROMPT",
    "REDIRECTION_PROMPT",
    "get_tutor_prompt",
    "format_tutor_context",
    "format_student_message",
    "format_hint_request",
    "format_verification_request",
]
