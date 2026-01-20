"""
Validador pedagógico.

Este módulo valida que las respuestas del tutor cumplan con los
principios pedagógicos del método socrático.
"""

from __future__ import annotations

import re

from config.settings import GuardrailsConfig, get_settings
from src.core.exceptions import PedagogicalValidationError
from src.core.types import GuardrailResult, HintLevel
from src.guardrails.base import BaseGuardrail, GuardrailCheckResult, GuardrailContext
from src.guardrails.patterns import (
    DIRECTIVE_LANGUAGE_PATTERNS,
    QUESTION_INDICATORS,
    compile_patterns,
    contains_question,
    count_questions,
    get_question_ratio,
)


class PedagogicalValidator(BaseGuardrail):
    """
    Validador de calidad pedagógica.

    Valida:
    - Presencia de preguntas (método socrático)
    - Progresión correcta de pistas
    - Lenguaje de guía vs. directivo
    - Relevancia al problema

    Attributes:
        config: Configuración de guardrails.
        min_question_ratio: Ratio mínimo de preguntas requerido.
    """

    def __init__(
        self,
        config: GuardrailsConfig | None = None,
    ) -> None:
        """
        Inicializa el validador pedagógico.

        Args:
            config: Configuración de guardrails.
        """
        self.config = config or get_settings().guardrails
        super().__init__(
            name="pedagogical_validator",
            enabled=self.config.pedagogical_validation_enabled,
        )

        self.min_question_ratio = self.config.min_question_ratio

        # Pre-compilar patrones
        self._question_patterns = compile_patterns(
            QUESTION_INDICATORS, "questions"
        )
        self._directive_patterns = compile_patterns(
            DIRECTIVE_LANGUAGE_PATTERNS, "directive"
        )

    async def check(self, context: GuardrailContext) -> GuardrailCheckResult:
        """
        Valida la calidad pedagógica de la respuesta del tutor.

        Args:
            context: Contexto con la respuesta a validar.

        Returns:
            Resultado de la validación.

        Raises:
            PedagogicalValidationError: Si falla la validación crítica.
        """
        if not self.is_enabled():
            return self._create_pass_result(
                reason="Validador pedagógico deshabilitado",
            )

        response = context.tutor_response
        if not response:
            return self._create_pass_result(
                reason="No hay respuesta para validar",
            )

        # Ejecutar validaciones
        has_questions, question_ratio = self._check_question_presence(response)
        proper_progression, progression_issue = self._check_hint_level_progression(
            response,
            context.current_hint_level,
            context.session.current_hint_level if context.session else None,
        )
        guiding_score, directive_phrases = self._check_guiding_language(response)
        topic_score, is_on_topic = self._check_topic_relevance(response, context)

        # Calcular score pedagógico
        pedagogical_score, metrics = self._calculate_pedagogical_score(
            has_questions=has_questions,
            question_ratio=question_ratio,
            proper_hint_progression=proper_progression,
            guiding_score=guiding_score,
            topic_relevance=topic_score,
        )

        # Preparar detalles
        details = {
            "question_analysis": {
                "has_questions": has_questions,
                "question_ratio": question_ratio,
                "question_count": count_questions(response),
                "min_required_ratio": self.min_question_ratio,
            },
            "hint_progression": {
                "proper": proper_progression,
                "issue": progression_issue,
                "current_level": context.current_hint_level.value,
            },
            "language_analysis": {
                "guiding_score": guiding_score,
                "directive_phrases_found": directive_phrases[:3],
            },
            "topic_relevance": {
                "score": topic_score,
                "is_on_topic": is_on_topic,
            },
            "overall_metrics": metrics,
        }

        # Determinar resultado
        # Fallo crítico: sin preguntas Y respuesta larga
        if not has_questions and len(response) > 100:
            return self._create_block_result(
                reason="Respuesta sin preguntas guía (método socrático requerido)",
                score=1.0 - pedagogical_score,
                suggested_action="Añadir preguntas que guíen al estudiante",
                validation_issue="missing_questions",
                **details,
            )

        # Fallo crítico: lenguaje muy directivo
        if guiding_score < 0.3:
            return self._create_block_result(
                reason="Respuesta demasiado directiva (falta método socrático)",
                score=1.0 - guiding_score,
                suggested_action="Reformular usando preguntas en lugar de instrucciones",
                validation_issue="directive_language",
                **details,
            )

        # Advertencia: ratio de preguntas bajo
        if question_ratio < self.min_question_ratio:
            return self._create_warn_result(
                reason=f"Ratio de preguntas bajo ({question_ratio:.0%} < {self.min_question_ratio:.0%})",
                score=1.0 - question_ratio,
                suggested_action="Añadir más preguntas para guiar al estudiante",
                validation_issue="low_question_ratio",
                **details,
            )

        # Advertencia: progresión de hints incorrecta
        if not proper_progression:
            return self._create_warn_result(
                reason=f"Progresión de pistas incorrecta: {progression_issue}",
                score=0.6,
                suggested_action="Ajustar nivel de pista según la progresión",
                validation_issue="hint_progression",
                **details,
            )

        # Advertencia: off-topic
        if not is_on_topic:
            return self._create_warn_result(
                reason="Respuesta parece estar fuera del tema del problema",
                score=1.0 - topic_score,
                suggested_action="Redirigir respuesta al problema específico",
                validation_issue="off_topic",
                **details,
            )

        # Pasa validación
        return self._create_pass_result(
            reason="Respuesta cumple criterios pedagógicos",
            score=pedagogical_score,
            **details,
        )

    def _check_question_presence(self, response: str) -> tuple[bool, float]:
        """
        Verifica si la respuesta contiene preguntas guía.

        Args:
            response: Respuesta del tutor.

        Returns:
            Tupla (tiene_preguntas, ratio_de_preguntas).
        """
        has_questions = contains_question(response)
        question_ratio = get_question_ratio(response)

        return has_questions, question_ratio

    def _check_hint_level_progression(
        self,
        response: str,
        current_level: HintLevel,
        previous_level: HintLevel | None,
    ) -> tuple[bool, str | None]:
        """
        Verifica que la progresión de pistas sea correcta.

        Args:
            response: Respuesta del tutor.
            current_level: Nivel de pista actual.
            previous_level: Nivel de pista anterior.

        Returns:
            Tupla (es_correcta, problema_detectado).
        """
        # Si no hay nivel anterior, cualquier nivel es válido
        if previous_level is None:
            return True, None

        # No permitir retroceder en nivel
        if current_level.value < previous_level.value:
            return False, f"Retroceso de nivel {previous_level.name} a {current_level.name}"

        # No permitir saltar niveles (SUBTLE a DIRECT directamente)
        if current_level.value - previous_level.value > 1:
            return False, f"Salto de nivel de {previous_level.name} a {current_level.name}"

        # Verificar si la respuesta es apropiada para el nivel
        response_lower = response.lower()

        if current_level == HintLevel.SUBTLE:
            # Nivel sutil: solo preguntas muy abiertas
            direct_indicators = [
                "primero", "first", "debes", "you should",
                "el siguiente paso", "next step",
            ]
            for indicator in direct_indicators:
                if indicator in response_lower:
                    return False, "Respuesta muy directa para nivel SUBTLE"

        elif current_level == HintLevel.MODERATE:
            # Nivel moderado: puede dar dirección pero no respuesta
            answer_indicators = [
                "la respuesta es", "the answer is",
                "el resultado es", "the result is",
                "= ", "igual a",
            ]
            for indicator in answer_indicators:
                if indicator in response_lower:
                    return False, "Respuesta demasiado directa para nivel MODERATE"

        # DIRECT permite respuestas más directas pero aún no la solución

        return True, None

    def _check_guiding_language(self, response: str) -> tuple[float, list[str]]:
        """
        Verifica el uso de lenguaje guía vs. directivo.

        Args:
            response: Respuesta del tutor.

        Returns:
            Tupla (score_de_guía 0-1, frases_directivas_encontradas).
        """
        directive_phrases: list[str] = []
        response_lower = response.lower()

        # Buscar patrones directivos
        for pattern in self._directive_patterns:
            matches = pattern.findall(response_lower)
            if matches:
                directive_phrases.extend(matches)

        # Contar oraciones
        sentences = re.split(r"[.!?¿]\s*", response)
        sentences = [s.strip() for s in sentences if s.strip()]
        total_sentences = max(len(sentences), 1)

        # Calcular score (más alto = más guía, menos directivo)
        directive_ratio = len(directive_phrases) / total_sentences
        guiding_score = max(0.0, 1.0 - directive_ratio)

        # Bonus por tener preguntas
        if contains_question(response):
            guiding_score = min(1.0, guiding_score + 0.2)

        return guiding_score, directive_phrases

    def _check_topic_relevance(
        self,
        response: str,
        context: GuardrailContext,
    ) -> tuple[float, bool]:
        """
        Verifica la relevancia de la respuesta al problema.

        Args:
            response: Respuesta del tutor.
            context: Contexto con información del problema.

        Returns:
            Tupla (score_de_relevancia, está_en_tema).
        """
        if context.solution is None:
            # Sin solución para comparar, asumir on-topic
            return 1.0, True

        solution = context.solution
        response_lower = response.lower()

        # Verificar mención de conceptos del problema
        concept_mentions = 0
        for concept in solution.concepts:
            if concept.lower() in response_lower:
                concept_mentions += 1

        # Verificar relevancia al tipo de problema
        problem_type_keywords = {
            "mathematics": ["número", "number", "ecuación", "equation", "suma", "add",
                          "resta", "subtract", "multiplica", "multiply", "divide",
                          "variable", "x", "y", "="],
            "physics": ["fuerza", "force", "velocidad", "velocity", "masa", "mass",
                       "energía", "energy", "gravedad", "gravity"],
            "chemistry": ["elemento", "element", "molécula", "molecule", "reacción",
                         "reaction", "átomo", "atom"],
            "programming": ["código", "code", "función", "function", "variable",
                           "loop", "bucle", "if", "else"],
        }

        type_keywords = problem_type_keywords.get(solution.problem_type.value, [])
        keyword_mentions = sum(
            1 for kw in type_keywords if kw in response_lower
        )

        # Calcular score
        concept_score = min(concept_mentions / max(len(solution.concepts), 1), 1.0)
        keyword_score = min(keyword_mentions / max(len(type_keywords), 1), 1.0)

        relevance_score = (concept_score * 0.6 + keyword_score * 0.4)

        # Es on-topic si el score es mayor a 0.3
        is_on_topic = relevance_score >= 0.3

        return relevance_score, is_on_topic

    def _calculate_pedagogical_score(
        self,
        has_questions: bool,
        question_ratio: float,
        proper_hint_progression: bool,
        guiding_score: float,
        topic_relevance: float,
    ) -> tuple[float, dict]:
        """
        Calcula el score pedagógico general.

        Args:
            has_questions: Si tiene preguntas.
            question_ratio: Ratio de preguntas.
            proper_hint_progression: Si la progresión es correcta.
            guiding_score: Score de lenguaje guía.
            topic_relevance: Score de relevancia al tema.

        Returns:
            Tupla (score_total, métricas_detalladas).
        """
        # Pesos para cada componente
        weights = {
            "questions": 0.35,
            "hint_progression": 0.20,
            "guiding_language": 0.25,
            "topic_relevance": 0.20,
        }

        # Scores individuales
        question_score = question_ratio if has_questions else 0.0
        progression_score = 1.0 if proper_hint_progression else 0.5

        # Score ponderado
        total_score = (
            question_score * weights["questions"] +
            progression_score * weights["hint_progression"] +
            guiding_score * weights["guiding_language"] +
            topic_relevance * weights["topic_relevance"]
        )

        metrics = {
            "question_score": question_score,
            "progression_score": progression_score,
            "guiding_score": guiding_score,
            "topic_relevance_score": topic_relevance,
            "weighted_total": total_score,
            "weights": weights,
        }

        return total_score, metrics


# =============================================================================
# Exports
# =============================================================================

__all__ = ["PedagogicalValidator"]
