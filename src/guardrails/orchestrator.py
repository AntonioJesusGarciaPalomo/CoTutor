"""
Orquestador del sistema de guardrails.

Este módulo coordina todos los detectores y filtros del sistema
de guardrails en un pipeline unificado.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.settings import GuardrailsConfig, get_settings
from src.core.types import (
    GuardrailResult,
    HintLevel,
    StudentInput,
    StructuredSolution,
    TutorResponse,
    TutoringSession,
)
from src.guardrails.base import GuardrailCheckResult, GuardrailContext
from src.guardrails.detectors.manipulation import ManipulationDetector
from src.guardrails.detectors.pedagogical import PedagogicalValidator
from src.guardrails.detectors.solution_leak import SolutionLeakDetector
from src.guardrails.filters.input_filter import InputFilter
from src.guardrails.filters.response_filter import ResponseFilter
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.models.base import BaseModelAdapter


class GuardrailsOrchestrator:
    """
    Orquestador del sistema de guardrails.

    Coordina la validación de entrada del estudiante y salida del tutor,
    aplicando detectores y filtros según configuración.

    Attributes:
        config: Configuración de guardrails.
        manipulation_detector: Detector de manipulación.
        solution_leak_detector: Detector de fugas de solución.
        pedagogical_validator: Validador pedagógico.
        input_filter: Filtro de entrada.
        response_filter: Filtro de respuesta.
    """

    def __init__(
        self,
        config: GuardrailsConfig | None = None,
        embedding_adapter: "BaseModelAdapter | None" = None,
    ) -> None:
        """
        Inicializa el orquestador.

        Args:
            config: Configuración de guardrails.
            embedding_adapter: Adaptador para embeddings (opcional).
        """
        self.config = config or get_settings().guardrails
        self.logger = get_logger("guardrail.orchestrator")

        # Inicializar detectores
        self.manipulation_detector = ManipulationDetector(self.config)
        self.solution_leak_detector = SolutionLeakDetector(
            self.config,
            embedding_adapter,
        )
        self.pedagogical_validator = PedagogicalValidator(self.config)

        # Inicializar filtros
        self.input_filter = InputFilter()
        self.response_filter = ResponseFilter()

        self.logger.info(
            "orchestrator_initialized",
            manipulation_enabled=self.config.manipulation_detection_enabled,
            leak_detection_enabled=self.config.solution_leak_detection_enabled,
            pedagogical_enabled=self.config.pedagogical_validation_enabled,
        )

    @classmethod
    async def create(
        cls,
        config: GuardrailsConfig | None = None,
        embedding_adapter: "BaseModelAdapter | None" = None,
    ) -> "GuardrailsOrchestrator":
        """
        Factory method para crear un orquestador configurado.

        Args:
            config: Configuración de guardrails.
            embedding_adapter: Adaptador para embeddings.

        Returns:
            Instancia configurada del orquestador.
        """
        return cls(config, embedding_adapter)

    async def validate_input(
        self,
        raw_input: str,
        session: TutoringSession | None = None,
    ) -> tuple[StudentInput, GuardrailResult]:
        """
        Valida y procesa la entrada del estudiante.

        Pipeline:
        1. Filtrar y sanitizar input
        2. Ejecutar detector de manipulación
        3. Clasificar intención
        4. Retornar input procesado con resultado

        Args:
            raw_input: Input crudo del estudiante.
            session: Sesión de tutoría actual (opcional).

        Returns:
            Tupla (StudentInput procesado, resultado del guardrail).
        """
        self.logger.debug(
            "validating_input",
            input_length=len(raw_input),
            has_session=session is not None,
        )

        # Paso 1: Filtrar input
        student_input = self.input_filter.filter(raw_input)

        # Paso 2: Crear contexto
        context = GuardrailContext(
            student_input=student_input,
            raw_input=raw_input,
            session=session,
            current_hint_level=session.current_hint_level if session else HintLevel.SUBTLE,
        )

        # Paso 3: Ejecutar detector de manipulación
        try:
            manipulation_result = await self.manipulation_detector.check(context)

            # Actualizar student_input con resultados
            student_input.manipulation_score = manipulation_result.score
            student_input.detected_intent = self.manipulation_detector.classify_intent(
                raw_input,
                manipulation_result.score,
                manipulation_result.details.get("detected_intent", "unknown"),
            )

        except Exception as e:
            self.logger.error(
                "manipulation_detection_failed",
                error=str(e),
            )
            manipulation_result = GuardrailCheckResult(
                result=GuardrailResult.PASS,
                score=0.0,
                reason=f"Error en detección: {e}",
            )

        # Paso 4: Determinar resultado final
        final_result = manipulation_result.result

        self.logger.info(
            "input_validated",
            result=final_result.value,
            manipulation_score=student_input.manipulation_score,
            detected_intent=student_input.detected_intent.value,
        )

        return student_input, final_result

    async def validate_response(
        self,
        response: str,
        solution: StructuredSolution,
        session: TutoringSession,
    ) -> TutorResponse:
        """
        Valida y filtra la respuesta del tutor.

        Pipeline:
        1. Ejecutar detector de fugas de solución
        2. Ejecutar validador pedagógico
        3. Aplicar filtros según resultados
        4. Retornar TutorResponse con resultados

        Args:
            response: Respuesta del tutor.
            solution: Solución estructurada del problema.
            session: Sesión de tutoría actual.

        Returns:
            TutorResponse con contenido validado/filtrado.
        """
        self.logger.debug(
            "validating_response",
            response_length=len(response),
            problem_type=solution.problem_type.value,
        )

        # Crear contexto
        context = GuardrailContext(
            tutor_response=response,
            solution=solution,
            session=session,
            current_hint_level=session.current_hint_level,
        )

        # Diccionario para almacenar resultados
        check_results: dict[str, GuardrailCheckResult] = {}

        # Ejecutar detector de fugas
        try:
            leak_result = await self.solution_leak_detector.check(context)
            check_results["solution_leak_detector"] = leak_result
        except Exception as e:
            self.logger.error(
                "leak_detection_failed",
                error=str(e),
            )
            check_results["solution_leak_detector"] = GuardrailCheckResult(
                result=GuardrailResult.PASS,
                score=0.0,
                reason=f"Error en detección: {e}",
            )

        # Ejecutar validador pedagógico
        try:
            pedagogical_result = await self.pedagogical_validator.check(context)
            check_results["pedagogical_validator"] = pedagogical_result
        except Exception as e:
            self.logger.error(
                "pedagogical_validation_failed",
                error=str(e),
            )
            check_results["pedagogical_validator"] = GuardrailCheckResult(
                result=GuardrailResult.PASS,
                score=0.0,
                reason=f"Error en validación: {e}",
            )

        # Aplicar filtros si es necesario
        filtered_response = response
        was_modified = False

        if self._should_filter(check_results):
            filtered_response, was_modified = self.response_filter.filter(
                response,
                solution,
                check_results,
                session.current_hint_level,
            )

        # Agregar resultados al diccionario de guardrails
        guardrail_results = {
            name: result.result
            for name, result in check_results.items()
        }

        # Crear TutorResponse
        tutor_response = TutorResponse(
            content=filtered_response,
            contains_question="?" in filtered_response or "¿" in filtered_response,
            hint_level_used=session.current_hint_level,
            guardrail_results=guardrail_results,
            was_modified=was_modified,
            original_content=response if was_modified else None,
        )

        # Determinar resultado agregado
        aggregated_result = self._aggregate_results(list(check_results.values()))

        self.logger.info(
            "response_validated",
            aggregated_result=aggregated_result.value,
            was_modified=was_modified,
            checks_performed=list(check_results.keys()),
        )

        return tutor_response

    async def process_full_turn(
        self,
        student_input: str,
        tutor_response: str,
        solution: StructuredSolution,
        session: TutoringSession,
    ) -> tuple[StudentInput, TutorResponse]:
        """
        Procesa un turno completo de conversación.

        Args:
            student_input: Input del estudiante.
            tutor_response: Respuesta del tutor.
            solution: Solución del problema.
            session: Sesión de tutoría.

        Returns:
            Tupla (StudentInput validado, TutorResponse validada).
        """
        # Validar input del estudiante
        validated_input, input_result = await self.validate_input(
            student_input,
            session,
        )

        # Validar respuesta del tutor
        validated_response = await self.validate_response(
            tutor_response,
            solution,
            session,
        )

        return validated_input, validated_response

    def _should_filter(self, results: dict[str, GuardrailCheckResult]) -> bool:
        """
        Determina si se debe aplicar filtrado.

        Args:
            results: Resultados de los detectores.

        Returns:
            True si se debe filtrar, False si no.
        """
        for result in results.values():
            if not result.is_pass():
                return True
        return False

    def _aggregate_results(
        self,
        results: list[GuardrailCheckResult],
    ) -> GuardrailResult:
        """
        Agrega múltiples resultados en uno solo.

        Lógica:
        - Si alguno es BLOCK → BLOCK
        - Si alguno es WARN y ninguno BLOCK → WARN
        - Si todos son PASS → PASS

        Args:
            results: Lista de resultados a agregar.

        Returns:
            Resultado agregado.
        """
        if not results:
            return GuardrailResult.PASS

        # Verificar si hay algún BLOCK
        if any(r.is_block() for r in results):
            return GuardrailResult.BLOCK

        # Verificar si hay algún WARN
        if any(r.is_warn() for r in results):
            return GuardrailResult.WARN

        return GuardrailResult.PASS

    async def set_embedding_adapter(
        self,
        adapter: "BaseModelAdapter",
    ) -> None:
        """
        Configura el adaptador de embeddings para detección semántica.

        Args:
            adapter: Adaptador de modelo para embeddings.
        """
        await self.solution_leak_detector.set_embedding_adapter(adapter)
        self.logger.info("embedding_adapter_configured")

    def get_config(self) -> GuardrailsConfig:
        """
        Obtiene la configuración actual.

        Returns:
            Configuración de guardrails.
        """
        return self.config

    def get_status(self) -> dict:
        """
        Obtiene el estado actual del orquestador.

        Returns:
            Diccionario con estado de cada componente.
        """
        return {
            "manipulation_detector": {
                "enabled": self.manipulation_detector.is_enabled(),
                "threshold": self.config.manipulation_threshold,
            },
            "solution_leak_detector": {
                "enabled": self.solution_leak_detector.is_enabled(),
                "semantic_threshold": self.config.semantic_similarity_threshold,
                "key_answer_threshold": self.config.key_answer_match_threshold,
            },
            "pedagogical_validator": {
                "enabled": self.pedagogical_validator.is_enabled(),
                "min_question_ratio": self.config.min_question_ratio,
            },
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = ["GuardrailsOrchestrator"]
