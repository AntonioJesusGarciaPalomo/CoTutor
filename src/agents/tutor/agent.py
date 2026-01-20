"""
Agente Tutor (Tutor Agent).

Este agente guía a estudiantes usando el método socrático,
NUNCA revelando soluciones directamente.

El Tutor:
- Recibe la solución estructurada del Solver
- Guía al estudiante mediante preguntas
- Proporciona pistas progresivas
- Valida todas sus respuestas con Guardrails
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from src.core.types import (
    GuardrailResult,
    HintLevel,
    MessageRole,
    StudentInput,
    StructuredSolution,
    TutorResponse,
    TutoringSession,
)
from src.models.base import BaseModelAdapter
from src.models.factory import get_model
from src.utils.logging import get_logger
from src.utils.metrics import get_metrics

from .prompts import (
    format_hint_request,
    format_student_message,
    format_tutor_context,
    format_verification_request,
    get_tutor_prompt,
)
from .session_manager import SessionManager, session_manager
from .strategies import StrategySelector, TutoringStrategy, strategy_selector


logger = get_logger(__name__)
metrics = get_metrics()


class TutorAgent:
    """
    Agente Tutor que guía estudiantes usando método socrático.

    Este agente es la interfaz principal con el estudiante. Recibe
    preguntas y guía sin revelar la solución, usando el sistema de
    guardrails para asegurar que no se filtren respuestas.

    Attributes:
        model: Adaptador del modelo de lenguaje.
        guardrails: Orquestador de guardrails.
        session_manager: Gestor de sesiones.
        strategy_selector: Selector de estrategias.

    Example:
        ```python
        # Crear agente
        tutor = await TutorAgent.create("ollama/llama3.1:8b")

        # Iniciar sesión con solución del Solver
        session = await tutor.start_session(problem, solution)

        # Responder al estudiante
        response = await tutor.respond(
            session.session_id,
            "¿Cómo empiezo a resolver esto?"
        )
        print(response.content)  # Pregunta socrática, no respuesta
        ```
    """

    def __init__(
        self,
        model: BaseModelAdapter,
        guardrails: Any | None = None,
        session_mgr: SessionManager | None = None,
        strategy_sel: StrategySelector | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_retries: int = 2,
    ) -> None:
        """
        Inicializa el Tutor Agent.

        Args:
            model: Adaptador del modelo de lenguaje.
            guardrails: Orquestador de guardrails (opcional).
            session_mgr: Gestor de sesiones (opcional).
            strategy_sel: Selector de estrategias (opcional).
            temperature: Temperatura del modelo (mayor para variedad).
            max_tokens: Máximo de tokens a generar.
            max_retries: Máximo de reintentos en caso de error.
        """
        self.model = model
        self.guardrails = guardrails
        self.session_manager = session_mgr or session_manager
        self.strategy_selector = strategy_sel or strategy_selector
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        logger.info(
            "TutorAgent inicializado",
            model=model.model_id,
            has_guardrails=guardrails is not None,
        )

    @classmethod
    async def create(
        cls,
        model_id: str = "ollama/llama3.1:8b",
        use_guardrails: bool = True,
        **kwargs: Any,
    ) -> "TutorAgent":
        """
        Factory method para crear un TutorAgent.

        Args:
            model_id: Identificador del modelo a usar.
            use_guardrails: Si usar el sistema de guardrails.
            **kwargs: Argumentos adicionales para el constructor.

        Returns:
            TutorAgent configurado y listo.
        """
        model = await get_model(model_id)

        guardrails = None
        if use_guardrails:
            from src.guardrails.orchestrator import GuardrailsOrchestrator
            guardrails = await GuardrailsOrchestrator.create()

        return cls(model=model, guardrails=guardrails, **kwargs)

    async def start_session(
        self,
        problem_text: str,
        solution: StructuredSolution,
    ) -> TutoringSession:
        """
        Inicia una nueva sesión de tutoría.

        Args:
            problem_text: Texto del problema a trabajar.
            solution: Solución estructurada del Solver.

        Returns:
            Nueva sesión de tutoría.
        """
        session = await self.session_manager.create_session(
            problem_text=problem_text,
            solution=solution,
        )

        metrics.increment("tutor_sessions_started")

        logger.info(
            "Sesión de tutoría iniciada",
            session_id=str(session.session_id),
            problem_preview=problem_text[:50],
        )

        return session

    async def respond(
        self,
        session_id: UUID,
        student_message: str,
    ) -> TutorResponse:
        """
        Genera una respuesta al mensaje del estudiante.

        Pipeline:
        1. Validar input del estudiante con guardrails
        2. Seleccionar estrategia de tutoría
        3. Generar respuesta con el LLM
        4. Validar respuesta con guardrails
        5. Actualizar sesión

        Args:
            session_id: ID de la sesión de tutoría.
            student_message: Mensaje del estudiante.

        Returns:
            Respuesta del tutor validada.

        Raises:
            ValueError: Si la sesión no existe.
        """
        start_time = time.perf_counter()

        # Obtener sesión
        session = await self.session_manager.get_session(session_id)
        if session is None:
            raise ValueError(f"Sesión no encontrada: {session_id}")

        if session.solution is None:
            raise ValueError("La sesión no tiene solución asociada")

        # Paso 1: Validar input del estudiante
        student_input, input_result = await self._validate_student_input(
            student_message, session
        )

        # Si el input es bloqueado por manipulación severa
        if input_result == GuardrailResult.BLOCK:
            return self._create_redirection_response(session, student_input)

        # Paso 2: Seleccionar estrategia
        strategy = self.strategy_selector.select_strategy(
            student_input=student_input,
            session=session,
            solution=session.solution,
        )

        # Paso 3: Generar respuesta
        raw_response = await self._generate_response(
            session=session,
            student_input=student_input,
            strategy=strategy,
        )

        # Paso 4: Validar respuesta con guardrails
        tutor_response = await self._validate_and_filter_response(
            raw_response,
            session,
            strategy,
        )

        # Paso 5: Actualizar sesión
        await self.session_manager.update_session(
            session_id=session_id,
            student_input=student_input,
            tutor_response=tutor_response,
        )

        # Métricas
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        tutor_response.response_time_ms = elapsed_ms
        tutor_response.tutor_model = self.model.model_id

        metrics.observe("tutor_response_time_ms", elapsed_ms)
        metrics.increment("tutor_responses_generated", labels={"strategy": strategy.value})

        logger.info(
            "Respuesta generada",
            session_id=str(session_id),
            strategy=strategy.value,
            was_modified=tutor_response.was_modified,
            elapsed_ms=elapsed_ms,
        )

        return tutor_response

    async def _validate_student_input(
        self,
        raw_input: str,
        session: TutoringSession,
    ) -> tuple[StudentInput, GuardrailResult]:
        """
        Valida el input del estudiante usando guardrails.

        Args:
            raw_input: Input crudo del estudiante.
            session: Sesión de tutoría actual.

        Returns:
            Tupla (StudentInput procesado, resultado del guardrail).
        """
        if self.guardrails:
            return await self.guardrails.validate_input(raw_input, session)

        # Sin guardrails, crear StudentInput básico
        from src.core.types import StudentIntent

        return StudentInput(
            raw_content=raw_input,
            processed_content=raw_input.strip(),
            detected_intent=StudentIntent.LEGITIMATE_QUESTION,
            intent_confidence=0.5,
            manipulation_score=0.0,
            is_on_topic=True,
        ), GuardrailResult.PASS

    async def _generate_response(
        self,
        session: TutoringSession,
        student_input: StudentInput,
        strategy: TutoringStrategy,
    ) -> str:
        """
        Genera una respuesta usando el LLM.

        Args:
            session: Sesión de tutoría actual.
            student_input: Input procesado del estudiante.
            strategy: Estrategia de tutoría a usar.

        Returns:
            Respuesta generada (sin validar).
        """
        if session.solution is None:
            raise ValueError("La sesión no tiene solución asociada")

        # Construir prompts
        system_prompt = get_tutor_prompt(strategy.value)
        context = format_tutor_context(session, session.solution)
        student_msg = format_student_message(student_input)

        # Construir historial de mensajes para el modelo
        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n{context}"},
        ]

        # Añadir historial de conversación reciente (últimos 6 mensajes)
        recent = session.conversation.get_last_n(6)
        for msg in recent:
            if msg.role == MessageRole.USER:
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == MessageRole.ASSISTANT:
                messages.append({"role": "assistant", "content": msg.content})

        # Añadir mensaje actual del estudiante
        messages.append({"role": "user", "content": student_msg})

        # Generar respuesta
        with metrics.timer("tutor_model_call_ms"):
            response = await self.model.generate(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        return response.content

    async def _validate_and_filter_response(
        self,
        raw_response: str,
        session: TutoringSession,
        strategy: TutoringStrategy,
    ) -> TutorResponse:
        """
        Valida y filtra la respuesta con guardrails.

        Args:
            raw_response: Respuesta generada sin filtrar.
            session: Sesión de tutoría actual.
            strategy: Estrategia usada para generar la respuesta.

        Returns:
            TutorResponse validada y posiblemente modificada.
        """
        if self.guardrails and session.solution:
            tutor_response = await self.guardrails.validate_response(
                raw_response,
                session.solution,
                session,
            )
            tutor_response.strategy_used = strategy.value
            return tutor_response

        # Sin guardrails, crear TutorResponse básica
        return TutorResponse(
            content=raw_response,
            contains_question="?" in raw_response or "¿" in raw_response,
            hint_level_used=session.current_hint_level,
            strategy_used=strategy.value,
            guardrail_results={},
            was_modified=False,
        )

    def _create_redirection_response(
        self,
        session: TutoringSession,
        student_input: StudentInput,
    ) -> TutorResponse:
        """
        Crea una respuesta de redirección para inputs bloqueados.

        Args:
            session: Sesión de tutoría actual.
            student_input: Input del estudiante (bloqueado).

        Returns:
            Respuesta que redirige al estudiante.
        """
        content = (
            "Entiendo que quieres avanzar rápido, pero mi rol es ayudarte a "
            "aprender, no darte respuestas directas. ¿Qué parte del problema "
            "te gustaría explorar juntos?"
        )

        return TutorResponse(
            content=content,
            contains_question=True,
            hint_level_used=session.current_hint_level,
            strategy_used=TutoringStrategy.REDIRECTION.value,
            guardrail_results={"input_validation": GuardrailResult.BLOCK},
            was_modified=False,
        )

    async def get_hint(
        self,
        session_id: UUID,
        level: HintLevel | None = None,
    ) -> str:
        """
        Obtiene una pista para el estudiante.

        Si no se especifica nivel, usa el nivel actual de la sesión.

        Args:
            session_id: ID de la sesión.
            level: Nivel de pista deseado (opcional).

        Returns:
            Contenido de la pista.

        Raises:
            ValueError: Si la sesión no existe.
        """
        session = await self.session_manager.get_session(session_id)
        if session is None:
            raise ValueError(f"Sesión no encontrada: {session_id}")

        if session.solution is None:
            raise ValueError("La sesión no tiene solución asociada")

        # Determinar nivel
        hint_level = level or session.current_hint_level

        # Obtener pistas disponibles
        hints = session.solution.get_hints_for_level(hint_level)

        if not hints:
            return "Piensa en los conceptos que has aprendido. ¿Qué herramientas tienes disponibles?"

        # Usar la pista del nivel apropiado
        for hint in reversed(hints):
            if hint.level == hint_level:
                # Reformular la pista usando el LLM
                prompt = format_hint_request(hint_level, hint.content)
                messages = [
                    {"role": "system", "content": get_tutor_prompt("hint_giving")},
                    {"role": "user", "content": prompt},
                ]

                response = await self.model.generate(
                    messages,
                    temperature=0.5,
                    max_tokens=200,
                )

                # Avanzar nivel de pista si es necesario
                if hint_level.value < HintLevel.DIRECT.value:
                    await self.session_manager.advance_hint_level(session_id)

                return response.content

        # Fallback
        return hints[-1].content if hints else "¿En qué parte del problema te sientes atascado?"

    async def verify_student_answer(
        self,
        session_id: UUID,
        student_answer: str,
    ) -> tuple[bool, str]:
        """
        Verifica la respuesta del estudiante.

        Args:
            session_id: ID de la sesión.
            student_answer: Respuesta propuesta por el estudiante.

        Returns:
            Tupla (es_correcta, feedback).

        Raises:
            ValueError: Si la sesión no existe.
        """
        session = await self.session_manager.get_session(session_id)
        if session is None:
            raise ValueError(f"Sesión no encontrada: {session_id}")

        if session.solution is None:
            raise ValueError("La sesión no tiene solución asociada")

        # Verificar respuesta
        is_correct = await self.session_manager.check_student_solution(
            session_id,
            student_answer,
        )

        # Generar feedback usando el LLM
        prompt = format_verification_request(
            student_answer=student_answer,
            correct_answer=session.solution.final_answer,
            is_correct=is_correct,
        )

        messages = [
            {"role": "system", "content": get_tutor_prompt("verification")},
            {"role": "user", "content": prompt},
        ]

        response = await self.model.generate(
            messages,
            temperature=0.5,
            max_tokens=200,
        )

        # Validar feedback con guardrails si es necesario
        feedback = response.content
        if self.guardrails and not is_correct:
            validated = await self.guardrails.validate_response(
                feedback,
                session.solution,
                session,
            )
            feedback = validated.content

        metrics.increment(
            "tutor_answer_verifications",
            labels={"correct": str(is_correct).lower()},
        )

        return is_correct, feedback

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
        """
        metrics_data = await self.session_manager.end_session(session_id, reason)

        metrics.increment(
            "tutor_sessions_ended",
            labels={
                "reason": reason,
                "solved": str(metrics_data.get("student_reached_solution", False)).lower(),
            },
        )

        return metrics_data

    def get_model_info(self) -> dict[str, Any]:
        """Obtiene información del modelo usado."""
        return {
            "model_id": self.model.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "has_guardrails": self.guardrails is not None,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TutorAgent",
]
