"""
Clases base para el sistema de guardrails.

Este módulo define las abstracciones fundamentales que todos los
guardrails deben implementar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.core.types import (
    ConversationHistory,
    GuardrailResult,
    HintLevel,
    StudentInput,
    StructuredSolution,
    TutoringSession,
)
from src.utils.logging import get_logger, log_guardrail_result


# =============================================================================
# Resultado de Guardrail
# =============================================================================

@dataclass
class GuardrailCheckResult:
    """
    Resultado de una verificación de guardrail.

    Attributes:
        result: Resultado de la verificación (PASS, WARN, BLOCK).
        score: Score de confianza/severidad (0.0 a 1.0).
        reason: Razón legible del resultado.
        details: Información adicional para debugging.
        suggested_action: Acción sugerida si el resultado es BLOCK o WARN.
    """

    result: GuardrailResult
    score: float
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    suggested_action: str | None = None

    def is_pass(self) -> bool:
        """Verifica si el resultado es PASS."""
        return self.result == GuardrailResult.PASS

    def is_warn(self) -> bool:
        """Verifica si el resultado es WARN."""
        return self.result == GuardrailResult.WARN

    def is_block(self) -> bool:
        """Verifica si el resultado es BLOCK."""
        return self.result == GuardrailResult.BLOCK

    def to_dict(self) -> dict[str, Any]:
        """Convierte el resultado a diccionario."""
        return {
            "result": self.result.value,
            "score": self.score,
            "reason": self.reason,
            "details": self.details,
            "suggested_action": self.suggested_action,
        }


# =============================================================================
# Contexto de Guardrail
# =============================================================================

@dataclass
class GuardrailContext:
    """
    Contexto pasado a los guardrails para evaluación.

    Este contexto contiene toda la información necesaria para que
    un guardrail tome una decisión informada.

    Attributes:
        student_input: Input procesado del estudiante (para guardrails de entrada).
        raw_input: Input crudo sin procesar.
        tutor_response: Respuesta del tutor a validar (para guardrails de salida).
        solution: Solución estructurada del problema actual.
        session: Sesión de tutoría actual.
        conversation_history: Historial de la conversación.
        current_hint_level: Nivel de pista actual en la sesión.
        metadata: Metadata adicional para el contexto.
    """

    # Input del estudiante
    student_input: StudentInput | None = None
    raw_input: str | None = None

    # Output del tutor
    tutor_response: str | None = None

    # Contexto compartido
    solution: StructuredSolution | None = None
    session: TutoringSession | None = None
    conversation_history: ConversationHistory | None = None

    # Estado actual
    current_hint_level: HintLevel = HintLevel.SUBTLE

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_text_to_check(self) -> str | None:
        """
        Obtiene el texto principal a verificar.

        Para guardrails de entrada: raw_input o student_input.processed_content
        Para guardrails de salida: tutor_response
        """
        if self.tutor_response is not None:
            return self.tutor_response
        if self.raw_input is not None:
            return self.raw_input
        if self.student_input is not None:
            return self.student_input.processed_content
        return None


# =============================================================================
# Base Guardrail
# =============================================================================

class BaseGuardrail(ABC):
    """
    Clase base abstracta para todos los guardrails.

    Todos los guardrails deben heredar de esta clase e implementar
    el método `check()`.

    Attributes:
        name: Nombre único del guardrail.
        enabled: Si el guardrail está habilitado.
        logger: Logger estructurado para el guardrail.
    """

    def __init__(self, name: str, enabled: bool = True) -> None:
        """
        Inicializa el guardrail base.

        Args:
            name: Nombre único del guardrail.
            enabled: Si el guardrail está habilitado.
        """
        self.name = name
        self.enabled = enabled
        self.logger = get_logger(f"guardrail.{name}")

    @abstractmethod
    async def check(self, context: GuardrailContext) -> GuardrailCheckResult:
        """
        Realiza la verificación del guardrail.

        Args:
            context: Contexto con toda la información necesaria.

        Returns:
            Resultado de la verificación.
        """
        pass

    def is_enabled(self) -> bool:
        """Verifica si este guardrail está habilitado."""
        return self.enabled

    def _create_pass_result(
        self,
        reason: str = "Verificación pasada",
        score: float = 0.0,
        **details: Any,
    ) -> GuardrailCheckResult:
        """
        Crea un resultado PASS.

        Args:
            reason: Razón del resultado.
            score: Score (generalmente bajo para PASS).
            **details: Detalles adicionales.

        Returns:
            GuardrailCheckResult con resultado PASS.
        """
        return self._create_result(
            GuardrailResult.PASS,
            score,
            reason,
            **details,
        )

    def _create_warn_result(
        self,
        reason: str,
        score: float,
        suggested_action: str | None = None,
        **details: Any,
    ) -> GuardrailCheckResult:
        """
        Crea un resultado WARN.

        Args:
            reason: Razón de la advertencia.
            score: Score de severidad.
            suggested_action: Acción sugerida.
            **details: Detalles adicionales.

        Returns:
            GuardrailCheckResult con resultado WARN.
        """
        return self._create_result(
            GuardrailResult.WARN,
            score,
            reason,
            suggested_action=suggested_action,
            **details,
        )

    def _create_block_result(
        self,
        reason: str,
        score: float,
        suggested_action: str | None = None,
        **details: Any,
    ) -> GuardrailCheckResult:
        """
        Crea un resultado BLOCK.

        Args:
            reason: Razón del bloqueo.
            score: Score de severidad.
            suggested_action: Acción sugerida.
            **details: Detalles adicionales.

        Returns:
            GuardrailCheckResult con resultado BLOCK.
        """
        return self._create_result(
            GuardrailResult.BLOCK,
            score,
            reason,
            suggested_action=suggested_action,
            **details,
        )

    def _create_result(
        self,
        result: GuardrailResult,
        score: float,
        reason: str,
        suggested_action: str | None = None,
        **details: Any,
    ) -> GuardrailCheckResult:
        """
        Factory method para crear resultados consistentes.

        Args:
            result: Tipo de resultado.
            score: Score de confianza/severidad.
            reason: Razón del resultado.
            suggested_action: Acción sugerida.
            **details: Detalles adicionales.

        Returns:
            GuardrailCheckResult configurado.
        """
        check_result = GuardrailCheckResult(
            result=result,
            score=score,
            reason=reason,
            details=details,
            suggested_action=suggested_action,
        )

        # Log the result
        self._log_result(check_result)

        return check_result

    def _log_result(self, result: GuardrailCheckResult) -> None:
        """
        Registra el resultado del guardrail.

        Args:
            result: Resultado a registrar.
        """
        log_guardrail_result(
            logger=self.logger,
            guardrail_name=self.name,
            result=result.result.value,
            score=result.score,
            reason=result.reason,
            details=result.details,
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "GuardrailCheckResult",
    "GuardrailContext",
    "BaseGuardrail",
]
