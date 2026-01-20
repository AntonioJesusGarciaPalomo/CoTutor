"""
Detector de fugas de solución.

Este módulo detecta cuando las respuestas del tutor revelan
información sensible de la solución al estudiante.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from config.settings import GuardrailsConfig, get_settings
from src.core.exceptions import SolutionLeakDetectedError
from src.core.types import GuardrailResult, StructuredSolution
from src.guardrails.base import BaseGuardrail, GuardrailCheckResult, GuardrailContext
from src.guardrails.patterns import extract_numeric_values, normalize_text

if TYPE_CHECKING:
    from src.models.base import BaseModelAdapter


class SolutionLeakDetector(BaseGuardrail):
    """
    Detector de fugas de solución en respuestas del tutor.

    Detecta:
    - Valores clave expuestos (key_values)
    - Respuesta final revelada (final_answer)
    - Pasos críticos revelados (is_critical=True)
    - Similaridad semántica alta con pasos de solución

    Attributes:
        config: Configuración de guardrails.
        semantic_threshold: Umbral para similaridad semántica.
        key_answer_threshold: Umbral para match de respuesta clave.
        step_threshold: Umbral para revelación de pasos.
    """

    def __init__(
        self,
        config: GuardrailsConfig | None = None,
        embedding_adapter: "BaseModelAdapter | None" = None,
    ) -> None:
        """
        Inicializa el detector de fugas.

        Args:
            config: Configuración de guardrails.
            embedding_adapter: Adaptador para embeddings (opcional).
        """
        self.config = config or get_settings().guardrails
        super().__init__(
            name="solution_leak_detector",
            enabled=self.config.solution_leak_detection_enabled,
        )

        self.semantic_threshold = self.config.semantic_similarity_threshold
        self.key_answer_threshold = self.config.key_answer_match_threshold
        self.step_threshold = self.config.step_revelation_threshold

        self._embedding_adapter = embedding_adapter
        self._embedding_model = self.config.embedding_model

    async def check(self, context: GuardrailContext) -> GuardrailCheckResult:
        """
        Verifica la respuesta del tutor por fugas de solución.

        Args:
            context: Contexto con la respuesta y solución.

        Returns:
            Resultado de la verificación.

        Raises:
            SolutionLeakDetectedError: Si se detecta fuga crítica.
        """
        if not self.is_enabled():
            return self._create_pass_result(
                reason="Detector de fugas deshabilitado",
            )

        response = context.tutor_response
        solution = context.solution

        if not response:
            return self._create_pass_result(
                reason="No hay respuesta para verificar",
            )

        if not solution:
            return self._create_pass_result(
                reason="No hay solución para comparar",
            )

        # Ejecutar todas las verificaciones
        key_value_score, leaked_values = await self._check_key_value_leakage(
            response, solution
        )
        final_answer_score, answer_leaked = self._check_final_answer_leak(
            response, solution
        )
        step_score, leaked_steps = self._check_step_revelation(response, solution)

        # Verificación semántica (si hay adaptador disponible)
        semantic_score = 0.0
        similar_content = None
        if self._embedding_adapter is not None:
            semantic_score, similar_content = await self._check_semantic_similarity(
                response, solution
            )

        # Calcular score final (usar el máximo)
        scores = {
            "key_value": key_value_score,
            "final_answer": final_answer_score,
            "step_revelation": step_score,
            "semantic_similarity": semantic_score,
        }
        max_type = max(scores, key=scores.get)  # type: ignore
        max_score = scores[max_type]

        # Preparar detalles
        details = {
            "key_value_leak": {
                "score": key_value_score,
                "leaked_values": leaked_values,
            },
            "final_answer_leak": {
                "score": final_answer_score,
                "leaked": answer_leaked,
            },
            "step_revelation": {
                "score": step_score,
                "leaked_steps": leaked_steps,
            },
            "semantic_similarity": {
                "score": semantic_score,
                "similar_content": similar_content,
            },
            "thresholds": {
                "semantic": self.semantic_threshold,
                "key_answer": self.key_answer_threshold,
                "step": self.step_threshold,
            },
        }

        # Determinar resultado
        # Zero tolerance para respuesta final
        if answer_leaked:
            result = self._create_block_result(
                reason="Respuesta final revelada en el contenido",
                score=1.0,
                suggested_action="Regenerar respuesta sin incluir la respuesta final",
                leak_type="final_answer",
                **details,
            )
            raise SolutionLeakDetectedError(
                response_preview=response[:100],
                leak_type="final_answer",
                similarity_score=1.0,
            )

        # Zero tolerance para key values
        if key_value_score > self.key_answer_threshold and leaked_values:
            result = self._create_block_result(
                reason=f"Valores clave revelados: {leaked_values[:3]}",
                score=key_value_score,
                suggested_action="Regenerar respuesta sin incluir valores numéricos clave",
                leak_type="key_values",
                **details,
            )
            raise SolutionLeakDetectedError(
                response_preview=response[:100],
                leak_type="key_values",
                similarity_score=key_value_score,
            )

        # Bloquear si revela pasos críticos
        if step_score >= self.step_threshold and leaked_steps:
            return self._create_block_result(
                reason=f"Pasos críticos revelados: {leaked_steps}",
                score=step_score,
                suggested_action="Regenerar respuesta sin revelar pasos críticos",
                leak_type="critical_steps",
                **details,
            )

        # Advertir por similaridad semántica alta
        if semantic_score >= self.semantic_threshold:
            return self._create_warn_result(
                reason="Alta similaridad semántica con solución",
                score=semantic_score,
                suggested_action="Reformular respuesta para ser menos directa",
                leak_type="semantic_similarity",
                **details,
            )

        # Pasa si no hay fugas significativas
        return self._create_pass_result(
            reason="No se detectaron fugas de solución",
            score=max_score,
            **details,
        )

    async def _check_key_value_leakage(
        self,
        response: str,
        solution: StructuredSolution,
    ) -> tuple[float, list[str]]:
        """
        Verifica si la respuesta contiene valores clave de la solución.

        Args:
            response: Respuesta del tutor.
            solution: Solución estructurada.

        Returns:
            Tupla (score, lista de valores filtrados).
        """
        if not solution.key_values:
            return 0.0, []

        leaked: list[str] = []
        response_normalized = normalize_text(response)

        # Extraer valores numéricos de la respuesta
        response_values = extract_numeric_values(response)

        for key_value in solution.key_values:
            key_normalized = normalize_text(str(key_value))

            # Verificar match exacto
            if key_normalized in response_normalized:
                leaked.append(key_value)
                continue

            # Verificar si el valor numérico está en la respuesta
            if key_value in response_values:
                leaked.append(key_value)
                continue

            # Verificar variantes comunes (con/sin decimales)
            try:
                num_value = float(key_value)
                # Verificar como entero si es número entero
                if num_value == int(num_value):
                    if str(int(num_value)) in response_values:
                        leaked.append(key_value)
            except ValueError:
                pass

        # Calcular score basado en proporción de valores filtrados
        if not leaked:
            return 0.0, []

        score = len(leaked) / len(solution.key_values)
        return min(score, 1.0), leaked

    def _check_final_answer_leak(
        self,
        response: str,
        solution: StructuredSolution,
    ) -> tuple[float, bool]:
        """
        Verifica si la respuesta contiene la respuesta final.

        Args:
            response: Respuesta del tutor.
            solution: Solución estructurada.

        Returns:
            Tupla (score, si fue filtrada).
        """
        if not solution.final_answer:
            return 0.0, False

        response_normalized = normalize_text(response)
        answer_normalized = normalize_text(solution.final_answer)

        # Verificar match exacto
        if answer_normalized in response_normalized:
            return 1.0, True

        # Verificar patrones comunes de respuesta
        answer_patterns = [
            rf"=\s*{re.escape(answer_normalized)}",
            rf"es\s+{re.escape(answer_normalized)}",
            rf"is\s+{re.escape(answer_normalized)}",
            rf"resultado\s*[:=]\s*{re.escape(answer_normalized)}",
            rf"answer\s*[:=]\s*{re.escape(answer_normalized)}",
            rf"respuesta\s*[:=]\s*{re.escape(answer_normalized)}",
        ]

        for pattern in answer_patterns:
            if re.search(pattern, response_normalized, re.IGNORECASE):
                return 1.0, True

        # Verificar si la respuesta numérica está presente
        try:
            # Extraer valor numérico de la respuesta final
            answer_match = re.search(r"-?\d+\.?\d*", solution.final_answer)
            if answer_match:
                answer_value = answer_match.group()
                if answer_value in response:
                    # Verificar contexto - ¿está en posición de "respuesta"?
                    context_patterns = [
                        rf"(=|es|is|igual\s+a)\s*{re.escape(answer_value)}",
                        rf"(resultado|answer|respuesta|solution)\s*[:=]?\s*{re.escape(answer_value)}",
                        rf"x\s*=\s*{re.escape(answer_value)}",
                    ]
                    for ctx_pattern in context_patterns:
                        if re.search(ctx_pattern, response, re.IGNORECASE):
                            return 1.0, True
        except Exception:
            pass

        return 0.0, False

    def _check_step_revelation(
        self,
        response: str,
        solution: StructuredSolution,
    ) -> tuple[float, list[int]]:
        """
        Verifica si la respuesta revela pasos críticos.

        Args:
            response: Respuesta del tutor.
            solution: Solución estructurada.

        Returns:
            Tupla (score, lista de números de pasos revelados).
        """
        critical_steps = solution.get_critical_steps()
        if not critical_steps:
            return 0.0, []

        response_normalized = normalize_text(response)
        leaked_steps: list[int] = []

        for step in critical_steps:
            # Verificar si el cálculo está en la respuesta
            if step.calculation:
                calc_normalized = normalize_text(step.calculation)
                if calc_normalized in response_normalized:
                    leaked_steps.append(step.step_number)
                    continue

            # Verificar si el resultado del paso está en la respuesta
            if step.result:
                result_normalized = normalize_text(step.result)
                if result_normalized in response_normalized:
                    leaked_steps.append(step.step_number)
                    continue

            # Verificar descripción muy similar
            if step.description:
                desc_words = set(normalize_text(step.description).split())
                response_words = set(response_normalized.split())
                # Si más del 70% de palabras de descripción están en respuesta
                if len(desc_words) > 3:
                    overlap = len(desc_words & response_words) / len(desc_words)
                    if overlap > 0.7:
                        leaked_steps.append(step.step_number)

        # Calcular score
        if not leaked_steps:
            return 0.0, []

        score = len(leaked_steps) / len(critical_steps)
        return min(score, 1.0), leaked_steps

    async def _check_semantic_similarity(
        self,
        response: str,
        solution: StructuredSolution,
    ) -> tuple[float, str | None]:
        """
        Verifica similaridad semántica entre respuesta y solución.

        Args:
            response: Respuesta del tutor.
            solution: Solución estructurada.

        Returns:
            Tupla (score máximo de similaridad, contenido similar encontrado).
        """
        if self._embedding_adapter is None:
            return 0.0, None

        try:
            # Preparar textos para comparación
            texts_to_compare = []

            # Añadir descripción de pasos críticos
            for step in solution.get_critical_steps():
                if step.description:
                    texts_to_compare.append(step.description)
                if step.calculation:
                    texts_to_compare.append(step.calculation)

            # Añadir respuesta final
            if solution.final_answer:
                texts_to_compare.append(f"La respuesta es {solution.final_answer}")

            if not texts_to_compare:
                return 0.0, None

            # Obtener embeddings
            all_texts = [response] + texts_to_compare
            embeddings = await self._get_embeddings(all_texts)

            if len(embeddings) < 2:
                return 0.0, None

            response_embedding = embeddings[0]
            solution_embeddings = embeddings[1:]

            # Calcular similaridad coseno con cada texto
            max_similarity = 0.0
            most_similar_text = None

            for i, sol_embedding in enumerate(solution_embeddings):
                similarity = self._cosine_similarity(response_embedding, sol_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_text = texts_to_compare[i]

            return max_similarity, most_similar_text

        except Exception as e:
            self.logger.warning(
                "semantic_similarity_failed",
                error=str(e),
            )
            return 0.0, None

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Obtiene embeddings para una lista de textos.

        Args:
            texts: Lista de textos.

        Returns:
            Lista de vectores de embedding.
        """
        if self._embedding_adapter is None:
            return []

        try:
            response = await self._embedding_adapter.embed(texts)
            return response.embeddings
        except Exception as e:
            self.logger.warning(
                "embedding_generation_failed",
                error=str(e),
            )
            return []

    def _cosine_similarity(
        self,
        vec1: list[float],
        vec2: list[float],
    ) -> float:
        """
        Calcula la similaridad coseno entre dos vectores.

        Args:
            vec1: Primer vector.
            vec2: Segundo vector.

        Returns:
            Similaridad coseno (0.0 a 1.0).
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def set_embedding_adapter(
        self,
        adapter: "BaseModelAdapter",
    ) -> None:
        """
        Configura el adaptador de embeddings.

        Args:
            adapter: Adaptador de modelo para embeddings.
        """
        self._embedding_adapter = adapter


# =============================================================================
# Exports
# =============================================================================

__all__ = ["SolutionLeakDetector"]
