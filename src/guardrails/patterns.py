"""
Patrones de detección para el sistema de guardrails.

Este módulo centraliza todos los patrones regex, keywords y constantes
utilizados para detectar manipulación, fugas de solución y validación pedagógica.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Pattern


# =============================================================================
# Patrones de Solicitud de Solución
# =============================================================================

SOLUTION_REQUEST_PATTERNS: list[str] = [
    # Inglés
    r"(just\s+)?give\s+me\s+the\s+(answer|solution|result)",
    r"tell\s+me\s+the\s+(answer|solution|result)",
    r"what\s+is\s+the\s+(final\s+)?(answer|solution|result)",
    r"show\s+me\s+the\s+(answer|solution|result)",
    r"i\s+(just\s+)?(want|need)\s+the\s+(answer|solution)",
    r"can\s+you\s+(just\s+)?solve\s+(it|this)",
    r"solve\s+(it|this)\s+for\s+me",
    r"what('s|\s+is)\s+the\s+value\s+of",
    r"calculate\s+(it|this)\s+for\s+me",
    # Español
    r"(solo\s+)?dame\s+la\s+(respuesta|soluci[oó]n|resultado)",
    r"dime\s+la\s+(respuesta|soluci[oó]n)",
    r"cu[aá]l\s+es\s+la\s+(respuesta|soluci[oó]n)",
    r"mu[eé]strame\s+la\s+(respuesta|soluci[oó]n)",
    r"quiero\s+la\s+(respuesta|soluci[oó]n)",
    r"necesito\s+la\s+(respuesta|soluci[oó]n)",
    r"resu[eé]lvelo\s+(t[uú]\s+)?por\s+m[ií]",
    r"calcula\s+(esto\s+)?por\s+m[ií]",
    r"hazlo\s+(t[uú]\s+)?por\s+m[ií]",
    r"no\s+quiero\s+pensar",
]


# =============================================================================
# Patrones de Prompt Injection
# =============================================================================

PROMPT_INJECTION_PATTERNS: list[str] = [
    # Inglés
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"forget\s+(everything|all|your\s+instructions)",
    r"disregard\s+(previous|all)\s+instructions",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*",
    r"\[system\]",
    r"<\s*system\s*>",
    r"override\s+(your\s+)?(instructions|programming)",
    r"you\s+must\s+(now\s+)?ignore",
    r"from\s+now\s+on\s+you\s+(will|must|should)",
    # Español
    r"ignora\s+(todas\s+)?(las\s+)?instrucciones",
    r"olvida\s+(todo|las\s+instrucciones)",
    r"nuevas\s+instrucciones\s*:",
    r"a\s+partir\s+de\s+ahora",
    r"desde\s+ahora\s+debes",
]


# =============================================================================
# Patrones de Jailbreak
# =============================================================================

JAILBREAK_PATTERNS: list[str] = [
    # Común
    r"dan\s+mode",
    r"developer\s+mode",
    r"jailbreak",
    r"no\s+restrictions",
    r"without\s+limitations",
    r"unrestricted\s+mode",
    r"evil\s+mode",
    r"god\s+mode",
    # Roleplay malicioso
    r"roleplay\s+as\s+(a\s+)?different",
    r"pretend\s+(you\s+are|to\s+be)\s+(a\s+)?different",
    r"act\s+as\s+if\s+you\s+(were|are)",
    r"you\s+are\s+now\s+a",
    r"imagine\s+you\s+are\s+no\s+longer",
    # Español
    r"modo\s+(desarrollador|sin\s+restricciones)",
    r"sin\s+limitaciones",
    r"finge\s+que\s+eres",
    r"actua\s+como\s+si",
    r"ahora\s+eres\s+(un|una)",
]


# =============================================================================
# Patrones de Bypass Socrático
# =============================================================================

SOCRATIC_BYPASS_PATTERNS: list[str] = [
    # Inglés
    r"stop\s+asking\s+(me\s+)?questions",
    r"don'?t\s+ask\s+(me\s+)?(any\s+)?questions",
    r"just\s+(tell|explain|show)",
    r"no\s+more\s+questions",
    r"i\s+don'?t\s+want\s+(to\s+)?think",
    r"don'?t\s+make\s+me\s+(think|work)",
    r"skip\s+the\s+(questions|hints)",
    r"get\s+to\s+the\s+point",
    r"enough\s+with\s+the\s+questions",
    # Español
    r"deja\s+de\s+preguntar(me)?",
    r"no\s+me\s+preguntes",
    r"solo\s+(dime|expl[ií]came|mu[eé]strame)",
    r"no\s+m[aá]s\s+preguntas",
    r"no\s+quiero\s+pensar",
    r"no\s+me\s+hagas\s+(pensar|trabajar)",
    r"salta(te)?\s+las\s+(preguntas|pistas)",
    r"ve\s+al\s+grano",
    r"basta\s+de\s+preguntas",
]


# =============================================================================
# Indicadores de Preguntas (para validación pedagógica)
# =============================================================================

QUESTION_INDICATORS: list[str] = [
    # Fin con signo de interrogación
    r"\?$",
    r"\?\s*$",
    # Inglés - inicio de pregunta
    r"^(what|how|why|when|where|which|who|whose|whom)\s",
    r"^(can|could|would|will|do|does|did|is|are|was|were|have|has|had)\s+(you|we|it|this|that)",
    r"^(think|consider|imagine|suppose)\s",
    # Español - inicio de pregunta
    r"^[¿]",
    r"^(qu[eé]|c[oó]mo|por\s+qu[eé]|cu[aá]ndo|d[oó]nde|cu[aá]l|qui[eé]n)\s",
    r"^(puedes|podr[ií]as|crees|piensas|consideras)\s",
    # Invitaciones a pensar
    r"(think\s+about|consider|piensa\s+en|considera)",
    r"(what\s+if|qu[eé]\s+pasar[ií]a\s+si)",
    r"(have\s+you\s+tried|has\s+intentado)",
    r"(what\s+do\s+you\s+think|qu[eé]\s+piensas|qu[eé]\s+crees)",
]


# =============================================================================
# Indicadores de Lenguaje Directivo (anti-socrático)
# =============================================================================

DIRECTIVE_LANGUAGE_PATTERNS: list[str] = [
    # Inglés - dar respuestas directas
    r"the\s+answer\s+is",
    r"the\s+solution\s+is",
    r"the\s+result\s+is",
    r"you\s+(need|have|must)\s+to",
    r"you\s+should\s+(just\s+)?",
    r"simply\s+(do|calculate|solve)",
    r"just\s+(do|calculate|solve|multiply|divide|add|subtract)",
    r"here'?s?\s+(the|your)\s+(answer|solution)",
    # Español - dar respuestas directas
    r"la\s+respuesta\s+es",
    r"la\s+soluci[oó]n\s+es",
    r"el\s+resultado\s+es",
    r"(debes|tienes\s+que|necesitas)",
    r"simplemente\s+(haz|calcula|resuelve)",
    r"solo\s+(haz|calcula|multiplica|divide|suma|resta)",
    r"aqu[ií]\s+(est[aá]|tienes)\s+(la|tu)\s+(respuesta|soluci[oó]n)",
]


# =============================================================================
# Keywords de Manipulación (para scoring)
# =============================================================================

MANIPULATION_KEYWORDS_EN: set[str] = {
    "answer", "solution", "result", "solve", "calculate",
    "tell me", "give me", "show me", "just", "directly",
    "skip", "ignore", "forget", "pretend", "bypass",
}

MANIPULATION_KEYWORDS_ES: set[str] = {
    "respuesta", "solucion", "solución", "resultado", "resolver", "calcular",
    "dime", "dame", "muestrame", "muéstrame", "solo", "directamente",
    "salta", "ignora", "olvida", "finge", "omite",
}

MANIPULATION_KEYWORDS: set[str] = MANIPULATION_KEYWORDS_EN | MANIPULATION_KEYWORDS_ES


# =============================================================================
# Patrones Numéricos
# =============================================================================

NUMERIC_PATTERNS: list[str] = [
    r"-?\d+\.?\d*",  # Números con decimales opcionales
    r"-?\d+/\d+",     # Fracciones
    r"-?\d+\s*%",     # Porcentajes
]

VARIABLE_ASSIGNMENT_PATTERNS: list[str] = [
    r"[a-zA-Z]\s*=\s*-?\d+\.?\d*",  # x = 5
    r"[a-zA-Z]\s*es\s+-?\d+\.?\d*",  # x es 5 (español)
]


# =============================================================================
# Funciones de Utilidad
# =============================================================================

_compiled_cache: dict[str, list[Pattern[str]]] = {}


def compile_patterns(patterns: list[str], cache_key: str | None = None) -> list[Pattern[str]]:
    """
    Compila una lista de patrones regex con flag case-insensitive.

    Args:
        patterns: Lista de patrones regex como strings.
        cache_key: Clave opcional para cachear los patrones compilados.

    Returns:
        Lista de patrones compilados.
    """
    if cache_key and cache_key in _compiled_cache:
        return _compiled_cache[cache_key]

    compiled = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns]

    if cache_key:
        _compiled_cache[cache_key] = compiled

    return compiled


def check_any_pattern(
    text: str,
    patterns: list[Pattern[str]],
) -> tuple[bool, str | None, float]:
    """
    Verifica si algún patrón coincide con el texto.

    Args:
        text: Texto a verificar.
        patterns: Lista de patrones compilados.

    Returns:
        Tupla (coincide, patrón_encontrado, score).
        El score es 1.0 si hay match, 0.0 si no.
    """
    for pattern in patterns:
        if pattern.search(text):
            return True, pattern.pattern, 1.0
    return False, None, 0.0


def count_pattern_matches(
    text: str,
    patterns: list[Pattern[str]],
) -> tuple[int, list[str]]:
    """
    Cuenta cuántos patrones coinciden con el texto.

    Args:
        text: Texto a verificar.
        patterns: Lista de patrones compilados.

    Returns:
        Tupla (cantidad_de_matches, lista_de_patrones_encontrados).
    """
    matches = []
    for pattern in patterns:
        if pattern.search(text):
            matches.append(pattern.pattern)
    return len(matches), matches


def extract_numeric_values(text: str) -> list[str]:
    """
    Extrae todos los valores numéricos del texto.

    Args:
        text: Texto del cual extraer valores.

    Returns:
        Lista de valores numéricos encontrados como strings.
    """
    patterns = compile_patterns(NUMERIC_PATTERNS, "numeric")
    values: list[str] = []

    for pattern in patterns:
        matches = pattern.findall(text)
        values.extend(matches)

    # Eliminar duplicados manteniendo orden
    seen: set[str] = set()
    unique_values: list[str] = []
    for v in values:
        normalized = v.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_values.append(normalized)

    return unique_values


def normalize_text(text: str) -> str:
    """
    Normaliza texto para comparación.

    - Convierte a minúsculas
    - Normaliza unicode (NFD → NFC)
    - Elimina espacios múltiples
    - Strip de espacios al inicio/fin

    Args:
        text: Texto a normalizar.

    Returns:
        Texto normalizado.
    """
    # Normalizar unicode
    text = unicodedata.normalize("NFC", text)
    # Minúsculas
    text = text.lower()
    # Espacios múltiples a uno solo
    text = re.sub(r"\s+", " ", text)
    # Strip
    text = text.strip()
    return text


def calculate_keyword_density(text: str, keywords: set[str]) -> float:
    """
    Calcula la densidad de keywords en el texto.

    Args:
        text: Texto a analizar.
        keywords: Set de keywords a buscar.

    Returns:
        Score de densidad (0.0 a 1.0).
    """
    normalized = normalize_text(text)
    words = normalized.split()

    if not words:
        return 0.0

    keyword_count = sum(1 for word in words if word in keywords)
    return min(keyword_count / len(words), 1.0)


def contains_question(text: str) -> bool:
    """
    Verifica si el texto contiene una pregunta.

    Args:
        text: Texto a verificar.

    Returns:
        True si contiene pregunta, False si no.
    """
    patterns = compile_patterns(QUESTION_INDICATORS, "questions")

    # Verificar cada línea/oración
    sentences = re.split(r"[.!]\s+", text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        for pattern in patterns:
            if pattern.search(sentence):
                return True

    return False


def count_questions(text: str) -> int:
    """
    Cuenta el número de preguntas en el texto.

    Args:
        text: Texto a analizar.

    Returns:
        Número de preguntas detectadas.
    """
    # Contar signos de interrogación como aproximación
    question_marks = text.count("?") + text.count("¿")
    return question_marks


def get_question_ratio(text: str) -> float:
    """
    Calcula el ratio de preguntas vs oraciones totales.

    Args:
        text: Texto a analizar.

    Returns:
        Ratio de preguntas (0.0 a 1.0).
    """
    # Dividir en oraciones
    sentences = re.split(r"[.!?¿]\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    question_count = count_questions(text)
    return min(question_count / len(sentences), 1.0)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Patrones
    "SOLUTION_REQUEST_PATTERNS",
    "PROMPT_INJECTION_PATTERNS",
    "JAILBREAK_PATTERNS",
    "SOCRATIC_BYPASS_PATTERNS",
    "QUESTION_INDICATORS",
    "DIRECTIVE_LANGUAGE_PATTERNS",
    "MANIPULATION_KEYWORDS",
    "MANIPULATION_KEYWORDS_EN",
    "MANIPULATION_KEYWORDS_ES",
    "NUMERIC_PATTERNS",
    "VARIABLE_ASSIGNMENT_PATTERNS",
    # Funciones
    "compile_patterns",
    "check_any_pattern",
    "count_pattern_matches",
    "extract_numeric_values",
    "normalize_text",
    "calculate_keyword_density",
    "contains_question",
    "count_questions",
    "get_question_ratio",
]
