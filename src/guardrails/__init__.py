"""
Sistema de Guardrails para Aula AI Tutor.

Este módulo proporciona un sistema completo de guardrails para:
- Detectar y prevenir manipulación por parte del estudiante
- Detectar y prevenir fugas de solución en respuestas del tutor
- Validar la calidad pedagógica de las respuestas (método socrático)
- Filtrar y modificar respuestas para cumplimiento pedagógico

Uso básico:
    from src.guardrails import GuardrailsOrchestrator

    # Crear orquestador
    orchestrator = await GuardrailsOrchestrator.create()

    # Validar input del estudiante
    student_input, result = await orchestrator.validate_input(
        "dame la respuesta directa"
    )

    # Validar respuesta del tutor
    tutor_response = await orchestrator.validate_response(
        "La respuesta es 5",
        solution=structured_solution,
        session=tutoring_session,
    )
"""

from src.guardrails.base import (
    BaseGuardrail,
    GuardrailCheckResult,
    GuardrailContext,
)
from src.guardrails.detectors.manipulation import ManipulationDetector
from src.guardrails.detectors.pedagogical import PedagogicalValidator
from src.guardrails.detectors.solution_leak import SolutionLeakDetector
from src.guardrails.filters.input_filter import InputFilter
from src.guardrails.filters.response_filter import ResponseFilter
from src.guardrails.orchestrator import GuardrailsOrchestrator


__all__ = [
    # Base classes
    "BaseGuardrail",
    "GuardrailCheckResult",
    "GuardrailContext",
    # Detectors
    "ManipulationDetector",
    "SolutionLeakDetector",
    "PedagogicalValidator",
    # Filters
    "InputFilter",
    "ResponseFilter",
    # Orchestrator
    "GuardrailsOrchestrator",
]
