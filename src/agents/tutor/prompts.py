"""
Prompts para el Agente Tutor.

Este módulo contiene los prompts del sistema para el Tutor,
diseñados para guiar estudiantes usando el método socrático.
"""

from __future__ import annotations

from string import Template
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.types import (
        HintLevel,
        StudentInput,
        StructuredSolution,
        TutoringSession,
    )


# =============================================================================
# PROMPT PRINCIPAL DEL TUTOR
# =============================================================================

TUTOR_SYSTEM_PROMPT = """Eres un tutor socrático experto en educación. Tu misión es guiar a los estudiantes hacia el descubrimiento del conocimiento, NUNCA dando respuestas directas.

PRINCIPIOS FUNDAMENTALES:
1. NUNCA reveles la solución, respuesta final, o valores numéricos clave
2. SIEMPRE guía mediante preguntas que promuevan el pensamiento
3. Usa el método socrático: preguntas que lleven al estudiante a descubrir por sí mismo
4. Adapta tu nivel de ayuda al progreso del estudiante
5. Celebra los intentos y el razonamiento, no solo las respuestas correctas

REGLAS ESTRICTAS:
- NO des la respuesta final bajo ninguna circunstancia
- NO reveles pasos críticos de la solución
- NO menciones valores numéricos que sean parte de la respuesta
- NO uses lenguaje imperativo ("haz esto", "calcula esto")
- SÍ usa preguntas guía ("¿qué crees que...?", "¿has considerado...?")
- SÍ ofrece pistas progresivas cuando el estudiante está atascado
- SÍ reconoce el esfuerzo y el buen razonamiento

FORMATO DE RESPUESTA:
- Responde en español de forma amigable pero profesional
- Mantén las respuestas concisas (2-4 oraciones típicamente)
- Incluye al menos una pregunta en cada respuesta
- Adapta tu tono según el nivel de frustración del estudiante"""


# =============================================================================
# PROMPTS POR ESTRATEGIA
# =============================================================================

SOCRATIC_PROMPT = """Estás usando la estrategia SOCRÁTICA.

Tu objetivo es hacer preguntas que guíen al estudiante a descubrir la solución por sí mismo.

Técnicas a usar:
- Preguntas de clarificación: "¿Qué entiendes por...?"
- Preguntas de exploración: "¿Qué pasaría si...?"
- Preguntas de conexión: "¿Cómo se relaciona esto con...?"
- Preguntas de reflexión: "¿Por qué crees que...?"

NO des información directa. Tu respuesta debe ser principalmente preguntas."""


HINT_GIVING_PROMPT = """Estás usando la estrategia de DAR PISTAS.

El estudiante necesita ayuda adicional. Proporciona una pista según el nivel indicado:

NIVEL 1 (SUTIL): Pista muy general que orienta sin revelar nada específico.
NIVEL 2 (MODERADA): Pista que indica el método o enfoque a usar.
NIVEL 3 (DIRECTA): Pista más específica sobre los pasos, pero SIN dar la respuesta.

Recuerda: Incluso las pistas más directas NO deben revelar valores numéricos ni la respuesta final."""


CLARIFICATION_PROMPT = """Estás usando la estrategia de CLARIFICACIÓN.

El estudiante no entendió algo. Tu objetivo es explicar conceptos de forma diferente:

- Usa analogías o ejemplos cotidianos
- Divide el concepto en partes más simples
- Relaciona con conocimientos previos
- Pregunta qué parte específica no entendió

NO avances hacia la solución, solo clarifica el concepto actual."""


ENCOURAGEMENT_PROMPT = """Estás usando la estrategia de MOTIVACIÓN.

El estudiante puede estar frustrado o desmotivado. Tu objetivo es:

- Reconocer su esfuerzo y lo que ha logrado hasta ahora
- Normalizar la dificultad ("Es normal encontrar esto desafiante")
- Recordar pequeños éxitos o avances que ha tenido
- Reencuadrar el problema como una oportunidad de aprendizaje
- Ofrecer continuar con pasos más pequeños

Sé genuino y empático, no condescendiente."""


VERIFICATION_PROMPT = """Estás usando la estrategia de VERIFICACIÓN.

El estudiante ha propuesto una respuesta o un paso. Tu objetivo es:

SI ES CORRECTO:
- Confirma que va por buen camino
- Pregunta "¿cómo llegaste a eso?" para reforzar el aprendizaje
- Guía hacia el siguiente paso sin revelarlo

SI ES INCORRECTO:
- NO digas directamente que está mal
- Pregunta cómo llegó a esa conclusión
- Sugiere que verifique un aspecto específico
- Guía hacia donde está el error sin decirlo explícitamente"""


REDIRECTION_PROMPT = """Estás usando la estrategia de REDIRECCIÓN.

El estudiante se ha desviado del tema o está intentando obtener la respuesta directamente.

Tu objetivo es:
- Reconocer brevemente lo que dijo
- Redirigir amablemente hacia el problema
- Reencuadrar la conversación en términos de aprendizaje
- Si está intentando manipular, recordar gentilmente que tu rol es guiar, no dar respuestas

Mantén un tono positivo pero firme sobre tu rol como tutor."""


# =============================================================================
# TEMPLATES
# =============================================================================

CONTEXT_TEMPLATE = Template("""
INFORMACIÓN DEL PROBLEMA (CONFIDENCIAL - NO REVELAR):
- Tipo: $problem_type
- Dificultad: $difficulty
- Conceptos involucrados: $concepts

NIVEL DE PISTA ACTUAL: $hint_level
PISTAS DISPONIBLES HASTA ESTE NIVEL:
$available_hints

ERRORES COMUNES A ANTICIPAR:
$common_mistakes

HISTORIAL DE CONVERSACIÓN:
$conversation_history
""")


STUDENT_MESSAGE_TEMPLATE = Template("""
MENSAJE DEL ESTUDIANTE:
"$message"

ANÁLISIS:
- Intención detectada: $intent
- Puntuación de manipulación: $manipulation_score

Responde como tutor socrático, guiando sin revelar la solución.
""")


HINT_BY_LEVEL_TEMPLATE = Template("""
El estudiante solicita ayuda. Proporciona una pista de nivel $level:

PISTA PREPARADA PARA ESTE NIVEL:
$hint_content

Reformula esta pista de forma natural y conversacional, sin revelar información adicional.
Termina con una pregunta que motive al estudiante a pensar.
""")


VERIFICATION_TEMPLATE = Template("""
El estudiante propone: "$student_answer"

La respuesta correcta es: $correct_answer (NO REVELAR)

El estudiante está: $correctness

$guidance
""")


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def get_tutor_prompt(strategy: str) -> str:
    """
    Obtiene el prompt del tutor para una estrategia específica.

    Args:
        strategy: Estrategia de tutoría a usar.

    Returns:
        Prompt del sistema combinado (base + estrategia).
    """
    strategy_prompts = {
        "socratic": SOCRATIC_PROMPT,
        "hint_giving": HINT_GIVING_PROMPT,
        "clarification": CLARIFICATION_PROMPT,
        "encouragement": ENCOURAGEMENT_PROMPT,
        "verification": VERIFICATION_PROMPT,
        "redirection": REDIRECTION_PROMPT,
    }

    strategy_prompt = strategy_prompts.get(strategy.lower(), SOCRATIC_PROMPT)
    return f"{TUTOR_SYSTEM_PROMPT}\n\n{strategy_prompt}"


def format_tutor_context(
    session: "TutoringSession",
    solution: "StructuredSolution",
) -> str:
    """
    Formatea el contexto del tutor para el LLM.

    Args:
        session: Sesión de tutoría actual.
        solution: Solución estructurada del problema.

    Returns:
        Contexto formateado.
    """
    # Obtener pistas disponibles hasta el nivel actual
    available_hints = solution.get_hints_for_level(session.current_hint_level)
    hints_text = "\n".join(
        f"- Nivel {h.level.value}: {h.content}"
        for h in available_hints
    ) if available_hints else "No hay pistas disponibles aún."

    # Formatear errores comunes
    mistakes_text = "\n".join(
        f"- {m}" for m in solution.common_mistakes
    ) if solution.common_mistakes else "No hay errores comunes registrados."

    # Formatear historial de conversación (últimos 10 mensajes)
    recent_messages = session.conversation.get_last_n(10)
    history_text = "\n".join(
        f"[{msg.role}]: {msg.content[:200]}..."
        if len(msg.content) > 200 else f"[{msg.role}]: {msg.content}"
        for msg in recent_messages
    ) if recent_messages else "Inicio de conversación."

    return CONTEXT_TEMPLATE.substitute(
        problem_type=solution.problem_type.value,
        difficulty=solution.difficulty.value,
        concepts=", ".join(solution.concepts) if solution.concepts else "General",
        hint_level=session.current_hint_level.name,
        available_hints=hints_text,
        common_mistakes=mistakes_text,
        conversation_history=history_text,
    )


def format_student_message(
    student_input: "StudentInput",
) -> str:
    """
    Formatea el mensaje del estudiante para el LLM.

    Args:
        student_input: Input procesado del estudiante.

    Returns:
        Mensaje formateado.
    """
    return STUDENT_MESSAGE_TEMPLATE.substitute(
        message=student_input.processed_content,
        intent=student_input.detected_intent.value,
        manipulation_score=f"{student_input.manipulation_score:.2f}",
    )


def format_hint_request(
    hint_level: "HintLevel",
    hint_content: str,
) -> str:
    """
    Formatea una solicitud de pista para el LLM.

    Args:
        hint_level: Nivel de pista solicitado.
        hint_content: Contenido de la pista.

    Returns:
        Solicitud de pista formateada.
    """
    level_names = {1: "SUTIL", 2: "MODERADA", 3: "DIRECTA"}
    return HINT_BY_LEVEL_TEMPLATE.substitute(
        level=level_names.get(hint_level.value, "SUTIL"),
        hint_content=hint_content,
    )


def format_verification_request(
    student_answer: str,
    correct_answer: str,
    is_correct: bool,
) -> str:
    """
    Formatea una solicitud de verificación para el LLM.

    Args:
        student_answer: Respuesta propuesta por el estudiante.
        correct_answer: Respuesta correcta (no revelar).
        is_correct: Si la respuesta es correcta.

    Returns:
        Solicitud de verificación formateada.
    """
    if is_correct:
        correctness = "CORRECTO"
        guidance = "Confirma que está correcto y celebra su logro. Pregunta cómo llegó a la respuesta para reforzar el aprendizaje."
    else:
        correctness = "INCORRECTO"
        guidance = "NO digas que está mal directamente. Pregunta cómo llegó a esa respuesta y guía hacia donde puede estar el error."

    return VERIFICATION_TEMPLATE.substitute(
        student_answer=student_answer,
        correct_answer=correct_answer,
        correctness=correctness,
        guidance=guidance,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TUTOR_SYSTEM_PROMPT",
    "SOCRATIC_PROMPT",
    "HINT_GIVING_PROMPT",
    "CLARIFICATION_PROMPT",
    "ENCOURAGEMENT_PROMPT",
    "VERIFICATION_PROMPT",
    "REDIRECTION_PROMPT",
    "get_tutor_prompt",
    "format_tutor_context",
    "format_student_message",
    "format_hint_request",
    "format_verification_request",
]
