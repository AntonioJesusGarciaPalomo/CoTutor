"""
Estrategias de tutoría para el Agente Tutor.

Este módulo define las diferentes estrategias que el tutor puede usar
para guiar al estudiante y la lógica de selección de estrategia.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.core.types import (
        StudentInput,
        StructuredSolution,
        TutoringSession,
    )


logger = get_logger(__name__)


class TutoringStrategy(str, Enum):
    """Estrategias de tutoría disponibles."""

    SOCRATIC = "socratic"
    """Usa preguntas para guiar al descubrimiento."""

    HINT_GIVING = "hint_giving"
    """Proporciona pistas según el nivel actual."""

    CLARIFICATION = "clarification"
    """Aclara conceptos que el estudiante no entiende."""

    VERIFICATION = "verification"
    """Verifica intentos de solución del estudiante."""

    ENCOURAGEMENT = "encouragement"
    """Motiva al estudiante frustrado o desmotivado."""

    REDIRECTION = "redirection"
    """Redirige cuando el estudiante se desvía o manipula."""


class StrategySelector:
    """
    Selector de estrategia de tutoría.

    Analiza el contexto de la conversación para determinar
    la mejor estrategia a usar en cada momento.
    """

    def __init__(self) -> None:
        """Inicializa el selector de estrategia."""
        self.logger = get_logger("tutor.strategy_selector")

    def select_strategy(
        self,
        student_input: "StudentInput",
        session: "TutoringSession",
        solution: "StructuredSolution",
    ) -> TutoringStrategy:
        """
        Selecciona la estrategia de tutoría más apropiada.

        La selección se basa en:
        1. Intención detectada del estudiante
        2. Nivel de manipulación detectado
        3. Estado actual de la sesión
        4. Progreso del estudiante

        Args:
            student_input: Input procesado del estudiante.
            session: Sesión de tutoría actual.
            solution: Solución estructurada del problema.

        Returns:
            Estrategia de tutoría a usar.
        """
        from src.core.types import StudentIntent

        intent = student_input.detected_intent
        manipulation_score = student_input.manipulation_score

        self.logger.debug(
            "selecting_strategy",
            intent=intent.value,
            manipulation_score=manipulation_score,
            hints_given=session.hints_given,
            hint_level=session.current_hint_level.value,
        )

        # Prioridad 1: Detectar manipulación
        if manipulation_score >= 0.7 or intent == StudentIntent.MANIPULATION_ATTEMPT:
            self.logger.info(
                "strategy_selected",
                strategy="redirection",
                reason="manipulation_detected",
            )
            return TutoringStrategy.REDIRECTION

        # Prioridad 2: Off-topic
        if intent == StudentIntent.OFF_TOPIC:
            self.logger.info(
                "strategy_selected",
                strategy="redirection",
                reason="off_topic",
            )
            return TutoringStrategy.REDIRECTION

        # Prioridad 3: Saludo (responder amigablemente pero redirigir)
        if intent == StudentIntent.GREETING:
            return TutoringStrategy.SOCRATIC

        # Prioridad 4: Intento de solución - verificar
        if intent == StudentIntent.SOLUTION_ATTEMPT:
            self.logger.info(
                "strategy_selected",
                strategy="verification",
                reason="solution_attempt",
            )
            return TutoringStrategy.VERIFICATION

        # Prioridad 5: Verificación explícita
        if intent == StudentIntent.VERIFICATION_REQUEST:
            return TutoringStrategy.VERIFICATION

        # Prioridad 6: Solicitud de pista
        if intent == StudentIntent.HINT_REQUEST:
            # Si ya ha pedido muchas pistas, considerar encouragement
            if session.hints_given >= 5 and manipulation_score >= 0.3:
                return TutoringStrategy.ENCOURAGEMENT
            return TutoringStrategy.HINT_GIVING

        # Prioridad 7: Clarificación
        if intent == StudentIntent.CLARIFICATION:
            return TutoringStrategy.CLARIFICATION

        # Prioridad 8: Pregunta legítima - usar método socrático
        if intent == StudentIntent.LEGITIMATE_QUESTION:
            # Considerar el progreso para ajustar la estrategia
            if self._student_seems_frustrated(session):
                return TutoringStrategy.ENCOURAGEMENT
            return TutoringStrategy.SOCRATIC

        # Default: Socrático
        self.logger.info(
            "strategy_selected",
            strategy="socratic",
            reason="default",
        )
        return TutoringStrategy.SOCRATIC

    def _student_seems_frustrated(
        self,
        session: "TutoringSession",
    ) -> bool:
        """
        Detecta si el estudiante parece frustrado.

        Indicadores:
        - Muchos turnos sin avance
        - Nivel de hint ya en DIRECT sin resolver
        - Mensajes cortos repetidos

        Args:
            session: Sesión de tutoría actual.

        Returns:
            True si el estudiante parece frustrado.
        """
        from src.core.types import HintLevel

        # Si ya está en nivel DIRECT y ha pedido muchas pistas
        if (
            session.current_hint_level == HintLevel.DIRECT
            and session.hints_given >= 3
        ):
            return True

        # Si hay muchos mensajes sin resolver
        if session.questions_asked >= 10 and not session.student_reached_solution:
            return True

        return False

    def get_strategy_description(self, strategy: TutoringStrategy) -> str:
        """
        Obtiene una descripción breve de la estrategia.

        Args:
            strategy: Estrategia de tutoría.

        Returns:
            Descripción de la estrategia.
        """
        descriptions = {
            TutoringStrategy.SOCRATIC: "Guiar mediante preguntas de descubrimiento",
            TutoringStrategy.HINT_GIVING: "Proporcionar pistas progresivas",
            TutoringStrategy.CLARIFICATION: "Aclarar conceptos no entendidos",
            TutoringStrategy.VERIFICATION: "Verificar intentos del estudiante",
            TutoringStrategy.ENCOURAGEMENT: "Motivar al estudiante",
            TutoringStrategy.REDIRECTION: "Redirigir hacia el problema",
        }
        return descriptions.get(strategy, "Estrategia no definida")


# Instancia global del selector
strategy_selector = StrategySelector()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TutoringStrategy",
    "StrategySelector",
    "strategy_selector",
]
