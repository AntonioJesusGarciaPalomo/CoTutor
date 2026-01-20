"""
Detector de manipulación del estudiante.

Este módulo detecta intentos de manipulación por parte del estudiante
para obtener soluciones directas, bypass del método socrático, o
ataques de prompt injection.
"""

from __future__ import annotations

from config.settings import GuardrailsConfig, get_settings
from src.core.exceptions import ManipulationDetectedError
from src.core.types import GuardrailResult, StudentIntent
from src.guardrails.base import BaseGuardrail, GuardrailCheckResult, GuardrailContext
from src.guardrails.patterns import (
    JAILBREAK_PATTERNS,
    MANIPULATION_KEYWORDS,
    PROMPT_INJECTION_PATTERNS,
    SOCRATIC_BYPASS_PATTERNS,
    SOLUTION_REQUEST_PATTERNS,
    calculate_keyword_density,
    check_any_pattern,
    compile_patterns,
    count_pattern_matches,
)


class ManipulationDetector(BaseGuardrail):
    """
    Detector de intentos de manipulación.

    Detecta:
    - Solicitudes directas de solución
    - Ataques de prompt injection
    - Intentos de jailbreak
    - Bypass del método socrático

    Attributes:
        threshold: Umbral para considerar manipulación (0.0 a 1.0).
        config: Configuración de guardrails.
    """

    def __init__(
        self,
        config: GuardrailsConfig | None = None,
    ) -> None:
        """
        Inicializa el detector de manipulación.

        Args:
            config: Configuración de guardrails. Si no se proporciona,
                   se usa la configuración global.
        """
        self.config = config or get_settings().guardrails
        super().__init__(
            name="manipulation_detector",
            enabled=self.config.manipulation_detection_enabled,
        )
        self.threshold = self.config.manipulation_threshold

        # Pre-compilar patrones
        self._solution_patterns = compile_patterns(
            SOLUTION_REQUEST_PATTERNS, "solution_request"
        )
        self._injection_patterns = compile_patterns(
            PROMPT_INJECTION_PATTERNS, "injection"
        )
        self._jailbreak_patterns = compile_patterns(
            JAILBREAK_PATTERNS, "jailbreak"
        )
        self._bypass_patterns = compile_patterns(
            SOCRATIC_BYPASS_PATTERNS, "bypass"
        )

    async def check(self, context: GuardrailContext) -> GuardrailCheckResult:
        """
        Verifica el input del estudiante por intentos de manipulación.

        Args:
            context: Contexto con el input a verificar.

        Returns:
            Resultado de la verificación.

        Raises:
            ManipulationDetectedError: Si la manipulación supera el umbral.
        """
        if not self.is_enabled():
            return self._create_pass_result(
                reason="Detector de manipulación deshabilitado",
            )

        text = context.get_text_to_check()
        if not text:
            return self._create_pass_result(
                reason="No hay texto para verificar",
            )

        # Ejecutar todas las detecciones
        solution_score, solution_details = self._detect_solution_request(text)
        injection_score, injection_details = self._detect_prompt_injection(text)
        jailbreak_score, jailbreak_details = self._detect_jailbreak(text)
        bypass_score, bypass_details = self._detect_socratic_bypass(text)

        # Calcular score combinado
        final_score, manipulation_type = self._calculate_manipulation_score(
            solution_score=solution_score,
            injection_score=injection_score,
            jailbreak_score=jailbreak_score,
            bypass_score=bypass_score,
        )

        # Clasificar intención
        intent = self.classify_intent(text, final_score, manipulation_type)

        # Preparar detalles
        details = {
            "solution_request": {
                "score": solution_score,
                "details": solution_details,
            },
            "prompt_injection": {
                "score": injection_score,
                "details": injection_details,
            },
            "jailbreak": {
                "score": jailbreak_score,
                "details": jailbreak_details,
            },
            "socratic_bypass": {
                "score": bypass_score,
                "details": bypass_details,
            },
            "detected_intent": intent.value,
            "threshold": self.threshold,
        }

        # Determinar resultado basado en score
        if final_score >= self.threshold:
            # Bloquear
            result = self._create_block_result(
                reason=f"Manipulación detectada: {manipulation_type}",
                score=final_score,
                suggested_action="Redirigir al estudiante hacia el método socrático",
                **details,
            )

            # Opcionalmente lanzar excepción
            if final_score >= 0.95:  # Muy alta confianza
                raise ManipulationDetectedError(
                    input_text=text,
                    manipulation_type=manipulation_type,
                    confidence=final_score,
                )

            return result

        elif final_score >= self.threshold * 0.7:  # ~0.56 con threshold=0.8
            # Advertencia
            return self._create_warn_result(
                reason=f"Posible intento de manipulación: {manipulation_type}",
                score=final_score,
                suggested_action="Monitorear y guiar hacia preguntas legítimas",
                **details,
            )

        else:
            # Pasa
            return self._create_pass_result(
                reason="No se detectó manipulación significativa",
                score=final_score,
                **details,
            )

    def _detect_solution_request(self, text: str) -> tuple[float, dict]:
        """
        Detecta solicitudes directas de solución.

        Args:
            text: Texto a analizar.

        Returns:
            Tupla (score, detalles).
        """
        matched, pattern, _ = check_any_pattern(text, self._solution_patterns)
        match_count, matched_patterns = count_pattern_matches(
            text, self._solution_patterns
        )

        # Score base por match
        score = 0.0
        if matched:
            score = 0.7  # Base alta para solicitud directa

        # Aumentar por múltiples matches
        if match_count > 1:
            score = min(score + 0.1 * (match_count - 1), 1.0)

        # Aumentar por keywords adicionales
        keyword_density = calculate_keyword_density(text, MANIPULATION_KEYWORDS)
        score = min(score + keyword_density * 0.2, 1.0)

        return score, {
            "matched": matched,
            "pattern": pattern,
            "match_count": match_count,
            "matched_patterns": matched_patterns[:3],  # Limitar para logs
        }

    def _detect_prompt_injection(self, text: str) -> tuple[float, dict]:
        """
        Detecta intentos de prompt injection.

        Args:
            text: Texto a analizar.

        Returns:
            Tupla (score, detalles).
        """
        matched, pattern, _ = check_any_pattern(text, self._injection_patterns)
        match_count, matched_patterns = count_pattern_matches(
            text, self._injection_patterns
        )

        # Score muy alto para injection (es un ataque)
        score = 0.0
        if matched:
            score = 0.9  # Muy alto para injection

        # Múltiples intentos = definitivamente malicioso
        if match_count > 1:
            score = min(score + 0.1, 1.0)

        return score, {
            "matched": matched,
            "pattern": pattern,
            "match_count": match_count,
        }

    def _detect_jailbreak(self, text: str) -> tuple[float, dict]:
        """
        Detecta intentos de jailbreak.

        Args:
            text: Texto a analizar.

        Returns:
            Tupla (score, detalles).
        """
        matched, pattern, _ = check_any_pattern(text, self._jailbreak_patterns)
        match_count, matched_patterns = count_pattern_matches(
            text, self._jailbreak_patterns
        )

        # Score muy alto para jailbreak (es un ataque)
        score = 0.0
        if matched:
            score = 0.95  # Máximo para jailbreak

        return score, {
            "matched": matched,
            "pattern": pattern,
            "match_count": match_count,
        }

    def _detect_socratic_bypass(self, text: str) -> tuple[float, dict]:
        """
        Detecta intentos de bypass del método socrático.

        Args:
            text: Texto a analizar.

        Returns:
            Tupla (score, detalles).
        """
        matched, pattern, _ = check_any_pattern(text, self._bypass_patterns)
        match_count, matched_patterns = count_pattern_matches(
            text, self._bypass_patterns
        )

        # Score moderado para bypass (frustración legítima posible)
        score = 0.0
        if matched:
            score = 0.5  # Moderado, puede ser frustración legítima

        # Múltiples intentos sugieren más intencionalidad
        if match_count > 1:
            score = min(score + 0.15 * (match_count - 1), 0.8)

        return score, {
            "matched": matched,
            "pattern": pattern,
            "match_count": match_count,
        }

    def _calculate_manipulation_score(
        self,
        solution_score: float,
        injection_score: float,
        jailbreak_score: float,
        bypass_score: float,
    ) -> tuple[float, str]:
        """
        Calcula el score combinado de manipulación.

        Args:
            solution_score: Score de solicitud de solución.
            injection_score: Score de prompt injection.
            jailbreak_score: Score de jailbreak.
            bypass_score: Score de bypass socrático.

        Returns:
            Tupla (score_final, tipo_de_manipulación).
        """
        # Usar el máximo score como base
        scores = {
            "solution_request": solution_score,
            "prompt_injection": injection_score,
            "jailbreak": jailbreak_score,
            "socratic_bypass": bypass_score,
        }

        max_type = max(scores, key=scores.get)  # type: ignore
        max_score = scores[max_type]

        # Añadir pequeño boost si hay múltiples tipos de manipulación
        active_types = sum(1 for s in scores.values() if s > 0.3)
        if active_types > 1:
            max_score = min(max_score + 0.1 * (active_types - 1), 1.0)

        return max_score, max_type

    def classify_intent(
        self,
        text: str,
        manipulation_score: float,
        manipulation_type: str,
    ) -> StudentIntent:
        """
        Clasifica la intención del estudiante basado en el análisis.

        Args:
            text: Texto del estudiante.
            manipulation_score: Score de manipulación calculado.
            manipulation_type: Tipo de manipulación detectado.

        Returns:
            Intención clasificada del estudiante.
        """
        # Si el score de manipulación es alto, es intento de manipulación
        if manipulation_score >= self.threshold:
            return StudentIntent.MANIPULATION_ATTEMPT

        # Si es moderado, podría ser frustración legítima
        if manipulation_score >= self.threshold * 0.5:
            if manipulation_type == "socratic_bypass":
                # Podría ser frustración, clasificar como hint request
                return StudentIntent.HINT_REQUEST
            return StudentIntent.MANIPULATION_ATTEMPT

        # Para scores bajos, usar clasificación basada en contenido
        text_lower = text.lower()

        # Verificar si es solicitud de pista
        hint_indicators = ["hint", "pista", "ayuda", "help", "stuck", "atascado"]
        if any(indicator in text_lower for indicator in hint_indicators):
            return StudentIntent.HINT_REQUEST

        # Verificar si es intento de solución
        solution_indicators = ["creo que", "i think", "my answer", "mi respuesta", "="]
        if any(indicator in text_lower for indicator in solution_indicators):
            return StudentIntent.SOLUTION_ATTEMPT

        # Verificar si pide verificación
        verification_indicators = ["correct", "correcto", "right", "bien", "check"]
        if any(indicator in text_lower for indicator in verification_indicators):
            return StudentIntent.VERIFICATION_REQUEST

        # Por defecto, pregunta legítima
        return StudentIntent.LEGITIMATE_QUESTION


# =============================================================================
# Exports
# =============================================================================

__all__ = ["ManipulationDetector"]
