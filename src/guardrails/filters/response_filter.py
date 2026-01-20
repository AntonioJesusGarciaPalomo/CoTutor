"""
Filtro de respuestas del tutor.

Este módulo modifica las respuestas del tutor para asegurar
el cumplimiento pedagógico cuando los detectores identifican problemas.
"""

from __future__ import annotations

import re
import random

from src.core.types import HintLevel, StructuredSolution
from src.guardrails.base import GuardrailCheckResult
from src.guardrails.patterns import (
    DIRECTIVE_LANGUAGE_PATTERNS,
    compile_patterns,
    extract_numeric_values,
    normalize_text,
)
from src.utils.logging import get_logger


class ResponseFilter:
    """
    Filtro y modificador de respuestas del tutor.

    Modifica respuestas para:
    - Enmascarar valores filtrados
    - Añadir preguntas guía si faltan
    - Ajustar nivel de pista
    - Reformular lenguaje directivo
    """

    # Plantillas de preguntas guía por nivel
    QUESTION_TEMPLATES = {
        HintLevel.SUBTLE: [
            "¿Qué crees que deberías hacer primero?",
            "¿Qué información tienes disponible?",
            "¿Puedes identificar qué tipo de problema es este?",
            "¿Qué conceptos crees que podrían ayudarte aquí?",
            "What do you think the first step might be?",
            "What information do you have available?",
        ],
        HintLevel.MODERATE: [
            "¿Qué operación podrías usar para resolver esto?",
            "¿Cómo podrías simplificar esta expresión?",
            "¿Qué pasa si intentas aislar la variable?",
            "¿Notas algún patrón en los números?",
            "What operation might help you here?",
            "Can you try to isolate the unknown?",
        ],
        HintLevel.DIRECT: [
            "Recuerda que puedes usar {concept} aquí.",
            "Piensa en cómo {operation} podría ayudarte.",
            "El siguiente paso sería considerar {hint}.",
            "Consider how {concept} applies here.",
            "The key insight involves {hint}.",
        ],
    }

    # Plantillas para reemplazar lenguaje directivo
    GUIDING_REPLACEMENTS = {
        "la respuesta es": "¿qué crees que podría ser la respuesta?",
        "el resultado es": "¿cuál piensas que es el resultado?",
        "debes hacer": "¿qué crees que podrías intentar?",
        "tienes que": "¿qué te parece si intentas?",
        "simplemente": "podrías considerar",
        "the answer is": "what do you think the answer might be?",
        "the result is": "what result do you get?",
        "you should": "what if you tried",
        "you need to": "consider trying",
        "just do": "you might try",
    }

    def __init__(self) -> None:
        """Inicializa el filtro de respuestas."""
        self.logger = get_logger("guardrail.response_filter")
        self._directive_patterns = compile_patterns(
            DIRECTIVE_LANGUAGE_PATTERNS, "directive"
        )

    def filter(
        self,
        response: str,
        solution: StructuredSolution,
        check_results: dict[str, GuardrailCheckResult],
        hint_level: HintLevel = HintLevel.SUBTLE,
    ) -> tuple[str, bool]:
        """
        Filtra y modifica la respuesta basado en resultados de guardrails.

        Args:
            response: Respuesta original del tutor.
            solution: Solución estructurada.
            check_results: Resultados de los detectores.
            hint_level: Nivel de pista actual.

        Returns:
            Tupla (respuesta_filtrada, fue_modificada).
        """
        modified = False
        filtered_response = response

        # 1. Enmascarar valores filtrados si hay leak
        if "solution_leak_detector" in check_results:
            leak_result = check_results["solution_leak_detector"]
            if not leak_result.is_pass():
                filtered_response, value_masked = self._mask_leaked_values(
                    filtered_response,
                    solution,
                    leak_result.details,
                )
                modified = modified or value_masked

        # 2. Añadir preguntas si faltan (validación pedagógica)
        if "pedagogical_validator" in check_results:
            ped_result = check_results["pedagogical_validator"]
            if not ped_result.is_pass():
                issue = ped_result.details.get("validation_issue", "")

                if issue in ("missing_questions", "low_question_ratio"):
                    filtered_response, questions_added = self._add_guiding_questions(
                        filtered_response,
                        solution,
                        hint_level,
                    )
                    modified = modified or questions_added

                if issue == "directive_language":
                    filtered_response, lang_modified = self._replace_directive_language(
                        filtered_response
                    )
                    modified = modified or lang_modified

        # 3. Asegurar que el nivel de pista sea apropiado
        filtered_response, level_adjusted = self._adjust_hint_level(
            filtered_response,
            hint_level,
        )
        modified = modified or level_adjusted

        if modified:
            self.logger.info(
                "response_filtered",
                original_length=len(response),
                filtered_length=len(filtered_response),
                modifications_applied=True,
            )

        return filtered_response, modified

    def _mask_leaked_values(
        self,
        response: str,
        solution: StructuredSolution,
        leak_details: dict,
    ) -> tuple[str, bool]:
        """
        Enmascara valores filtrados en la respuesta.

        Args:
            response: Respuesta a filtrar.
            solution: Solución con valores a proteger.
            leak_details: Detalles del leak detectado.

        Returns:
            Tupla (respuesta_filtrada, fue_modificada).
        """
        modified = False
        result = response

        # Obtener valores filtrados
        leaked_values = leak_details.get("key_value_leak", {}).get("leaked_values", [])
        answer_leaked = leak_details.get("final_answer_leak", {}).get("leaked", False)

        # Enmascarar respuesta final si fue filtrada
        if answer_leaked and solution.final_answer:
            result = self._mask_value(result, solution.final_answer)
            modified = True

        # Enmascarar key values filtrados
        for value in leaked_values:
            result = self._mask_value(result, value)
            modified = True

        # También enmascarar todos los key_values por seguridad
        for value in solution.key_values:
            if value in result:
                result = self._mask_value(result, value)
                modified = True

        return result, modified

    def _mask_value(self, text: str, value: str) -> str:
        """
        Enmascara un valor específico en el texto.

        Args:
            text: Texto donde enmascarar.
            value: Valor a enmascarar.

        Returns:
            Texto con valor enmascarado.
        """
        # Patrones donde el valor podría aparecer
        patterns = [
            # Valor en contexto de ecuación
            (rf"=\s*{re.escape(value)}(?!\d)", "= [?]"),
            # Valor como respuesta
            (rf"(?:es|is)\s+{re.escape(value)}(?!\d)", "es [?]"),
            # Valor directo (con límites de palabra)
            (rf"\b{re.escape(value)}\b", "[valor oculto]"),
        ]

        result = text
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    def _add_guiding_questions(
        self,
        response: str,
        solution: StructuredSolution,
        hint_level: HintLevel,
    ) -> tuple[str, bool]:
        """
        Añade preguntas guía a la respuesta.

        Args:
            response: Respuesta original.
            solution: Solución para contexto.
            hint_level: Nivel de pista actual.

        Returns:
            Tupla (respuesta_con_preguntas, fue_modificada).
        """
        # Verificar si ya tiene preguntas suficientes
        question_marks = response.count("?") + response.count("¿")
        if question_marks >= 2:
            return response, False

        # Obtener plantilla apropiada
        templates = self.QUESTION_TEMPLATES.get(hint_level, [])
        if not templates:
            templates = self.QUESTION_TEMPLATES[HintLevel.SUBTLE]

        # Seleccionar pregunta aleatoria
        question = random.choice(templates)

        # Personalizar pregunta si es nivel DIRECT
        if hint_level == HintLevel.DIRECT and solution.concepts:
            concept = random.choice(solution.concepts)
            question = question.replace("{concept}", concept)
            question = question.replace("{hint}", solution.hints[0].content if solution.hints else concept)
            question = question.replace("{operation}", "esta operación")

        # Añadir pregunta al final
        modified_response = response.rstrip()
        if not modified_response.endswith(("?", "¿")):
            modified_response += f"\n\n{question}"
            return modified_response, True

        return response, False

    def _replace_directive_language(self, response: str) -> tuple[str, bool]:
        """
        Reemplaza lenguaje directivo por lenguaje guía.

        Args:
            response: Respuesta a modificar.

        Returns:
            Tupla (respuesta_modificada, fue_modificada).
        """
        modified = False
        result = response

        for directive, guiding in self.GUIDING_REPLACEMENTS.items():
            if directive.lower() in result.lower():
                # Reemplazo case-insensitive
                pattern = re.compile(re.escape(directive), re.IGNORECASE)
                result = pattern.sub(guiding, result)
                modified = True

        return result, modified

    def _adjust_hint_level(
        self,
        response: str,
        target_level: HintLevel,
    ) -> tuple[str, bool]:
        """
        Ajusta la respuesta para coincidir con el nivel de pista esperado.

        Args:
            response: Respuesta a ajustar.
            target_level: Nivel de pista objetivo.

        Returns:
            Tupla (respuesta_ajustada, fue_modificada).
        """
        # Para nivel SUBTLE, remover información demasiado directa
        if target_level == HintLevel.SUBTLE:
            too_direct_patterns = [
                r"(el primer paso es|the first step is).*?[.!]",
                r"(deberías|you should).*?[.!]",
                r"(empieza por|start by).*?[.!]",
            ]

            modified = False
            result = response

            for pattern in too_direct_patterns:
                if re.search(pattern, result, re.IGNORECASE):
                    # Reemplazar por pregunta más sutil
                    result = re.sub(
                        pattern,
                        "¿Qué crees que sería un buen enfoque?",
                        result,
                        flags=re.IGNORECASE,
                    )
                    modified = True

            return result, modified

        return response, False

    def sanitize_for_student(
        self,
        response: str,
        solution: StructuredSolution,
    ) -> str:
        """
        Sanitiza completamente una respuesta para el estudiante.

        Este método aplica todas las protecciones necesarias
        independientemente de los resultados de los detectores.

        Args:
            response: Respuesta a sanitizar.
            solution: Solución con valores a proteger.

        Returns:
            Respuesta sanitizada.
        """
        result = response

        # Enmascarar respuesta final
        if solution.final_answer:
            result = self._mask_value(result, solution.final_answer)

        # Enmascarar todos los key_values
        for value in solution.key_values:
            result = self._mask_value(result, str(value))

        # Enmascarar cálculos de pasos críticos
        for step in solution.get_critical_steps():
            if step.calculation:
                # Extraer valores del cálculo
                values = extract_numeric_values(step.calculation)
                for value in values:
                    result = self._mask_value(result, value)
            if step.result:
                result = self._mask_value(result, step.result)

        return result


# =============================================================================
# Exports
# =============================================================================

__all__ = ["ResponseFilter"]
