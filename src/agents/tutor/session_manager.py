"""
Gestor de sesiones de tutoría.

Este módulo gestiona el ciclo de vida de las sesiones de tutoría,
incluyendo creación, actualización y finalización.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from uuid import UUID

from src.core.types import (
    ConversationHistory,
    HintLevel,
    Message,
    MessageRole,
    StudentInput,
    StructuredSolution,
    TutorResponse,
    TutoringSession,
)
from src.utils.logging import get_logger


logger = get_logger(__name__)


class SessionManager:
    """
    Gestor de sesiones de tutoría.

    Mantiene el estado de las sesiones activas y proporciona
    métodos para gestionar su ciclo de vida.

    Attributes:
        _sessions: Diccionario de sesiones activas.
        max_sessions: Máximo de sesiones concurrentes.
    """

    def __init__(self, max_sessions: int = 1000) -> None:
        """
        Inicializa el gestor de sesiones.

        Args:
            max_sessions: Máximo de sesiones concurrentes permitidas.
        """
        self._sessions: dict[UUID, TutoringSession] = {}
        self.max_sessions = max_sessions
        self.logger = get_logger("tutor.session_manager")

    async def create_session(
        self,
        problem_text: str,
        solution: StructuredSolution,
    ) -> TutoringSession:
        """
        Crea una nueva sesión de tutoría.

        Args:
            problem_text: Texto del problema.
            solution: Solución estructurada del Solver.

        Returns:
            Nueva sesión de tutoría.

        Raises:
            ValueError: Si se excede el límite de sesiones.
        """
        # Verificar límite de sesiones
        if len(self._sessions) >= self.max_sessions:
            # Limpiar sesiones antiguas
            self._cleanup_old_sessions()

            if len(self._sessions) >= self.max_sessions:
                raise ValueError(
                    f"Límite de sesiones alcanzado ({self.max_sessions})"
                )

        # Crear nueva sesión
        session = TutoringSession(
            problem_text=problem_text,
            solution=solution,
            conversation=ConversationHistory(),
            current_hint_level=HintLevel.SUBTLE,
            hints_given=0,
            questions_asked=0,
            started_at=datetime.now(),
        )

        # Añadir mensaje inicial del sistema
        session.conversation.add_message(
            role=MessageRole.SYSTEM,
            content=f"Sesión iniciada para problema: {problem_text[:100]}...",
        )

        # Guardar sesión
        self._sessions[session.session_id] = session

        self.logger.info(
            "session_created",
            session_id=str(session.session_id),
            problem_type=solution.problem_type.value,
            difficulty=solution.difficulty.value,
        )

        return session

    async def get_session(self, session_id: UUID) -> TutoringSession | None:
        """
        Obtiene una sesión por su ID.

        Args:
            session_id: ID de la sesión.

        Returns:
            Sesión de tutoría o None si no existe.
        """
        return self._sessions.get(session_id)

    async def update_session(
        self,
        session_id: UUID,
        student_input: StudentInput,
        tutor_response: TutorResponse,
    ) -> None:
        """
        Actualiza una sesión con un nuevo turno de conversación.

        Args:
            session_id: ID de la sesión.
            student_input: Input del estudiante.
            tutor_response: Respuesta del tutor.

        Raises:
            ValueError: Si la sesión no existe.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Sesión no encontrada: {session_id}")

        # Añadir mensaje del estudiante
        session.conversation.add_message(
            role=MessageRole.USER,
            content=student_input.processed_content,
            intent=student_input.detected_intent.value,
            manipulation_score=student_input.manipulation_score,
        )

        # Añadir respuesta del tutor
        session.conversation.add_message(
            role=MessageRole.ASSISTANT,
            content=tutor_response.content,
            strategy=tutor_response.strategy_used,
            hint_level=tutor_response.hint_level_used.value if tutor_response.hint_level_used else None,
            was_modified=tutor_response.was_modified,
        )

        # Actualizar contadores
        session.questions_asked += 1

        # Si se dio una pista, actualizar contador
        if tutor_response.hint_level_used:
            session.hints_given += 1

        self.logger.debug(
            "session_updated",
            session_id=str(session_id),
            questions_asked=session.questions_asked,
            hints_given=session.hints_given,
        )

    async def advance_hint_level(self, session_id: UUID) -> HintLevel:
        """
        Avanza el nivel de pista de una sesión.

        Args:
            session_id: ID de la sesión.

        Returns:
            Nuevo nivel de pista.

        Raises:
            ValueError: Si la sesión no existe.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Sesión no encontrada: {session_id}")

        previous_level = session.current_hint_level
        new_level = session.advance_hint_level()

        self.logger.info(
            "hint_level_advanced",
            session_id=str(session_id),
            previous_level=previous_level.value,
            new_level=new_level.value,
        )

        return new_level

    async def check_student_solution(
        self,
        session_id: UUID,
        student_answer: str,
    ) -> bool:
        """
        Verifica si la respuesta del estudiante es correcta.

        Args:
            session_id: ID de la sesión.
            student_answer: Respuesta propuesta por el estudiante.

        Returns:
            True si la respuesta es correcta.

        Raises:
            ValueError: Si la sesión no existe o no tiene solución.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Sesión no encontrada: {session_id}")

        if session.solution is None:
            raise ValueError("La sesión no tiene solución asociada")

        # Normalizar respuestas para comparación
        student_normalized = self._normalize_answer(student_answer)
        correct_normalized = self._normalize_answer(session.solution.final_answer)

        is_correct = student_normalized == correct_normalized

        if is_correct:
            session.student_reached_solution = True
            session.ended_at = datetime.now()
            self.logger.info(
                "student_solved_problem",
                session_id=str(session_id),
                questions_asked=session.questions_asked,
                hints_given=session.hints_given,
            )

        return is_correct

    def _normalize_answer(self, answer: str) -> str:
        """
        Normaliza una respuesta para comparación.

        Args:
            answer: Respuesta a normalizar.

        Returns:
            Respuesta normalizada.
        """
        # Convertir a minúsculas y eliminar espacios extra
        normalized = " ".join(answer.lower().split())

        # Eliminar signos de puntuación al final
        normalized = normalized.rstrip(".,;:")

        # Normalizar espacios alrededor de =
        normalized = re.sub(r'\s*=\s*', '=', normalized)

        # Eliminar espacios después de signos negativos
        normalized = re.sub(r'-\s+', '-', normalized)

        return normalized

    async def end_session(
        self,
        session_id: UUID,
        reason: str = "completed",
    ) -> dict[str, Any]:
        """
        Finaliza una sesión de tutoría.

        Args:
            session_id: ID de la sesión.
            reason: Razón de finalización.

        Returns:
            Métricas finales de la sesión.

        Raises:
            ValueError: Si la sesión no existe.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Sesión no encontrada: {session_id}")

        # Marcar como finalizada
        session.ended_at = datetime.now()

        # Calcular métricas
        metrics = self.get_session_metrics(session_id)
        metrics["end_reason"] = reason

        # Remover de sesiones activas
        del self._sessions[session_id]

        self.logger.info(
            "session_ended",
            session_id=str(session_id),
            reason=reason,
            duration_seconds=metrics.get("duration_seconds"),
            student_reached_solution=session.student_reached_solution,
        )

        return metrics

    def get_session_metrics(self, session_id: UUID) -> dict[str, Any]:
        """
        Obtiene métricas de una sesión.

        Args:
            session_id: ID de la sesión.

        Returns:
            Diccionario con métricas de la sesión.

        Raises:
            ValueError: Si la sesión no existe.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Sesión no encontrada: {session_id}")

        # Calcular duración
        end_time = session.ended_at or datetime.now()
        duration = (end_time - session.started_at).total_seconds()

        return {
            "session_id": str(session_id),
            "problem_type": session.solution.problem_type.value if session.solution else None,
            "difficulty": session.solution.difficulty.value if session.solution else None,
            "questions_asked": session.questions_asked,
            "hints_given": session.hints_given,
            "final_hint_level": session.current_hint_level.value,
            "student_reached_solution": session.student_reached_solution,
            "duration_seconds": duration,
            "started_at": session.started_at.isoformat(),
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "conversation_length": len(session.conversation.messages),
        }

    def _cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Limpia sesiones antiguas.

        Args:
            max_age_hours: Edad máxima en horas.

        Returns:
            Número de sesiones eliminadas.
        """
        from datetime import timedelta

        now = datetime.now()
        max_age = timedelta(hours=max_age_hours)
        removed = 0

        sessions_to_remove = [
            session_id
            for session_id, session in self._sessions.items()
            if (now - session.started_at) > max_age
        ]

        for session_id in sessions_to_remove:
            del self._sessions[session_id]
            removed += 1

        if removed > 0:
            self.logger.info(
                "old_sessions_cleaned",
                removed_count=removed,
            )

        return removed

    def get_active_sessions_count(self) -> int:
        """Obtiene el número de sesiones activas."""
        return len(self._sessions)

    def get_all_session_ids(self) -> list[UUID]:
        """Obtiene los IDs de todas las sesiones activas."""
        return list(self._sessions.keys())


# Instancia global del gestor de sesiones
session_manager = SessionManager()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SessionManager",
    "session_manager",
]
