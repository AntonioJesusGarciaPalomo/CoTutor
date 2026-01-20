"""
Filtro de entrada del estudiante.

Este módulo procesa y sanitiza el input del estudiante antes de
ser evaluado por los detectores.
"""

from __future__ import annotations

import re

from src.core.types import StudentInput, StudentIntent
from src.guardrails.patterns import (
    JAILBREAK_PATTERNS,
    MANIPULATION_KEYWORDS,
    PROMPT_INJECTION_PATTERNS,
    SOCRATIC_BYPASS_PATTERNS,
    SOLUTION_REQUEST_PATTERNS,
    calculate_keyword_density,
    check_any_pattern,
    compile_patterns,
    normalize_text,
)
from src.utils.logging import get_logger


class InputFilter:
    """
    Filtro y sanitizador de input del estudiante.

    Procesa el input crudo del estudiante para:
    - Normalizar y limpiar el texto
    - Remover marcadores de injection
    - Realizar clasificación inicial de intención
    """

    def __init__(self) -> None:
        """Inicializa el filtro de entrada."""
        self.logger = get_logger("guardrail.input_filter")

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

    def filter(self, raw_input: str) -> StudentInput:
        """
        Procesa el input crudo del estudiante.

        Args:
            raw_input: Input sin procesar del estudiante.

        Returns:
            StudentInput con el contenido procesado y clasificación inicial.
        """
        # Normalizar el input
        processed = self._normalize_input(raw_input)

        # Remover marcadores de injection
        processed = self._remove_injection_markers(processed)

        # Clasificar intención
        intent, confidence = self._classify_initial_intent(processed)

        # Calcular score de manipulación inicial
        manipulation_score = self._calculate_initial_manipulation_score(processed)

        # Verificar si está on-topic (básico - se refinará con el detector)
        is_on_topic = self._check_basic_on_topic(processed)

        return StudentInput(
            raw_content=raw_input,
            processed_content=processed,
            detected_intent=intent,
            intent_confidence=confidence,
            manipulation_score=manipulation_score,
            is_on_topic=is_on_topic,
        )

    def _normalize_input(self, text: str) -> str:
        """
        Normaliza el input del estudiante.

        Args:
            text: Texto a normalizar.

        Returns:
            Texto normalizado.
        """
        # Usar normalización base
        normalized = normalize_text(text)

        # Remover caracteres de control excepto newlines
        normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", normalized)

        # Limitar longitud excesiva (prevención básica)
        max_length = 10000
        if len(normalized) > max_length:
            normalized = normalized[:max_length]
            self.logger.warning(
                "input_truncated",
                original_length=len(text),
                max_length=max_length,
            )

        return normalized

    def _remove_injection_markers(self, text: str) -> str:
        """
        Remueve marcadores comunes de prompt injection.

        Args:
            text: Texto a limpiar.

        Returns:
            Texto sin marcadores de injection.
        """
        # Patrones de marcadores a remover
        markers_to_remove = [
            r"\[system\]",
            r"\[/system\]",
            r"<system>",
            r"</system>",
            r"\[assistant\]",
            r"\[/assistant\]",
            r"<assistant>",
            r"</assistant>",
            r"\[user\]",
            r"\[/user\]",
            r"<user>",
            r"</user>",
            r"###\s*(system|assistant|user)\s*:",
            r"```(system|assistant)?\s*\n",
            r"```\s*$",
        ]

        result = text
        for pattern in markers_to_remove:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)

        # Limpiar espacios múltiples resultantes
        result = re.sub(r"\s+", " ", result).strip()

        if result != text:
            self.logger.info(
                "injection_markers_removed",
                original_preview=text[:100],
                cleaned_preview=result[:100],
            )

        return result

    def _classify_initial_intent(self, text: str) -> tuple[StudentIntent, float]:
        """
        Realiza clasificación inicial de intención.

        Args:
            text: Texto procesado.

        Returns:
            Tupla (intención detectada, confianza).
        """
        # Verificar si es un saludo
        greeting_patterns = [
            r"^(hola|hi|hello|hey|buenos?\s+d[ií]as?|buenas?\s+(tardes?|noches?))",
            r"^(que\s+tal|how\s+are\s+you|c[oó]mo\s+est[aá]s?)",
        ]
        for pattern in greeting_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return StudentIntent.GREETING, 0.9

        # Verificar solicitud directa de solución (alta probabilidad de manipulación)
        matched, _, _ = check_any_pattern(text, self._solution_patterns)
        if matched:
            return StudentIntent.MANIPULATION_ATTEMPT, 0.85

        # Verificar prompt injection
        matched, _, _ = check_any_pattern(text, self._injection_patterns)
        if matched:
            return StudentIntent.MANIPULATION_ATTEMPT, 0.95

        # Verificar jailbreak
        matched, _, _ = check_any_pattern(text, self._jailbreak_patterns)
        if matched:
            return StudentIntent.MANIPULATION_ATTEMPT, 0.95

        # Verificar bypass socrático
        matched, _, _ = check_any_pattern(text, self._bypass_patterns)
        if matched:
            return StudentIntent.MANIPULATION_ATTEMPT, 0.75

        # Verificar si pide una pista
        hint_patterns = [
            r"(give|dame|necesito)\s+(me\s+)?(a\s+)?(hint|pista|ayuda)",
            r"(can\s+you\s+help|puedes\s+ayudarme)",
            r"(i'?m\s+stuck|estoy\s+atascado|no\s+s[eé]\s+c[oó]mo)",
            r"(don'?t\s+understand|no\s+entiendo|no\s+comprendo)",
        ]
        for pattern in hint_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return StudentIntent.HINT_REQUEST, 0.8

        # Verificar si es un intento de solución
        solution_attempt_patterns = [
            r"(i\s+think|creo\s+que|pienso\s+que)\s+.*(is|es|=)",
            r"(the\s+answer|la\s+respuesta)\s+(is|es|ser[ií]a)",
            r"(my\s+answer|mi\s+respuesta)",
            r"(would\s+it\s+be|ser[ií]a)\s+\d+",
            r"=\s*-?\d+",
        ]
        for pattern in solution_attempt_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return StudentIntent.SOLUTION_ATTEMPT, 0.7

        # Verificar si pide verificación
        verification_patterns = [
            r"(is\s+(this|that)\s+(correct|right)|est[aá]\s+(bien|correcto))",
            r"(did\s+i\s+(do|get)|lo\s+hice\s+bien)",
            r"(check|verifica|revisa)\s+(my|mi)",
        ]
        for pattern in verification_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return StudentIntent.VERIFICATION_REQUEST, 0.75

        # Verificar si pide clarificación
        clarification_patterns = [
            r"(what\s+do\s+you\s+mean|qu[eé]\s+quieres\s+decir)",
            r"(can\s+you\s+explain|puedes\s+explicar)",
            r"(i\s+don'?t\s+get|no\s+(entiendo|pillo))\s+(what|qu[eé])",
            r"(what\s+is|qu[eé]\s+es)\s+.+\?",
        ]
        for pattern in clarification_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return StudentIntent.CLARIFICATION, 0.7

        # Si tiene signo de interrogación, probablemente es pregunta legítima
        if "?" in text or "¿" in text:
            return StudentIntent.LEGITIMATE_QUESTION, 0.6

        # Por defecto, asumir pregunta legítima con baja confianza
        return StudentIntent.LEGITIMATE_QUESTION, 0.5

    def _calculate_initial_manipulation_score(self, text: str) -> float:
        """
        Calcula un score inicial de manipulación basado en keywords.

        Args:
            text: Texto a analizar.

        Returns:
            Score de manipulación (0.0 a 1.0).
        """
        # Densidad de keywords de manipulación
        keyword_density = calculate_keyword_density(text, MANIPULATION_KEYWORDS)

        # Verificar patrones de manipulación
        pattern_score = 0.0
        pattern_checks = [
            (self._solution_patterns, 0.3),
            (self._injection_patterns, 0.4),
            (self._jailbreak_patterns, 0.4),
            (self._bypass_patterns, 0.2),
        ]

        for patterns, weight in pattern_checks:
            matched, _, _ = check_any_pattern(text, patterns)
            if matched:
                pattern_score += weight

        # Combinar scores
        total_score = min(keyword_density * 0.3 + pattern_score, 1.0)

        return total_score

    def _check_basic_on_topic(self, text: str) -> bool:
        """
        Verificación básica de si el texto está on-topic.

        Esta es una verificación inicial que puede ser refinada
        por el detector con más contexto.

        Args:
            text: Texto a verificar.

        Returns:
            True si parece estar on-topic, False si no.
        """
        # Patrones claramente off-topic
        off_topic_patterns = [
            r"(weather|clima|tiempo)\s+(today|hoy|forecast)",
            r"(tell\s+me\s+a\s+joke|cu[eé]ntame\s+un\s+chiste)",
            r"(who\s+(are|is)\s+.*(president|rey|reina)|qui[eé]n\s+es)",
            r"(what\s+time|qu[eé]\s+hora)",
            r"(sports|deportes|f[uú]tbol|basketball)",
            r"(movie|pel[ií]cula|series?|netflix)",
            r"(music|m[uú]sica|canci[oó]n)",
            r"(food|comida|recipe|receta)",
        ]

        for pattern in off_topic_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False

        return True


# =============================================================================
# Exports
# =============================================================================

__all__ = ["InputFilter"]
