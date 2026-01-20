"""
Tests para los prompts del tutor.
"""

import pytest

from src.agents.tutor.prompts import (
    CLARIFICATION_PROMPT,
    ENCOURAGEMENT_PROMPT,
    HINT_GIVING_PROMPT,
    REDIRECTION_PROMPT,
    SOCRATIC_PROMPT,
    TUTOR_SYSTEM_PROMPT,
    VERIFICATION_PROMPT,
    format_hint_request,
    format_student_message,
    format_tutor_context,
    format_verification_request,
    get_tutor_prompt,
)
from src.core.types import (
    ConversationHistory,
    DifficultyLevel,
    Hint,
    HintLevel,
    Message,
    MessageRole,
    ProblemType,
    StudentInput,
    StudentIntent,
    StructuredSolution,
    TutoringSession,
)


@pytest.fixture
def sample_solution() -> StructuredSolution:
    """Crea una solución de ejemplo."""
    return StructuredSolution(
        problem_text="Resuelve: 2x + 3 = 7",
        problem_type=ProblemType.MATHEMATICS,
        difficulty=DifficultyLevel.BASIC,
        concepts=["ecuaciones", "álgebra"],
        steps=[],
        final_answer="x = 2",
        common_mistakes=["Olvidar cambiar signo", "Error al dividir"],
        hints=[
            Hint(level=HintLevel.SUBTLE, content="¿Qué operación necesitas?"),
            Hint(level=HintLevel.MODERATE, content="Intenta aislar x"),
        ],
    )


@pytest.fixture
def sample_session(sample_solution: StructuredSolution) -> TutoringSession:
    """Crea una sesión de ejemplo."""
    session = TutoringSession(
        problem_text="Resuelve: 2x + 3 = 7",
        solution=sample_solution,
        current_hint_level=HintLevel.SUBTLE,
    )
    # Añadir un mensaje al historial
    session.conversation.add_message(
        role=MessageRole.USER,
        content="¿Cómo empiezo?",
    )
    session.conversation.add_message(
        role=MessageRole.ASSISTANT,
        content="¿Qué operación crees que necesitas?",
    )
    return session


class TestTutorSystemPrompt:
    """Tests para el prompt del sistema."""

    def test_system_prompt_contains_key_instructions(self) -> None:
        """El prompt contiene instrucciones clave."""
        assert "NUNCA" in TUTOR_SYSTEM_PROMPT
        assert "socrático" in TUTOR_SYSTEM_PROMPT.lower()
        assert "preguntas" in TUTOR_SYSTEM_PROMPT.lower()

    def test_system_prompt_has_rules(self) -> None:
        """El prompt tiene reglas claras."""
        assert "NO" in TUTOR_SYSTEM_PROMPT
        assert "SÍ" in TUTOR_SYSTEM_PROMPT


class TestStrategyPrompts:
    """Tests para prompts de estrategias."""

    def test_socratic_prompt_focuses_on_questions(self) -> None:
        """El prompt socrático enfoca en preguntas."""
        assert "pregunta" in SOCRATIC_PROMPT.lower()
        assert "SOCRÁTICA" in SOCRATIC_PROMPT

    def test_hint_giving_prompt_has_levels(self) -> None:
        """El prompt de pistas tiene niveles."""
        assert "NIVEL 1" in HINT_GIVING_PROMPT
        assert "NIVEL 2" in HINT_GIVING_PROMPT
        assert "NIVEL 3" in HINT_GIVING_PROMPT

    def test_clarification_prompt_focuses_on_explanation(self) -> None:
        """El prompt de clarificación enfoca en explicación."""
        assert "CLARIFICACIÓN" in CLARIFICATION_PROMPT
        assert "explicar" in CLARIFICATION_PROMPT.lower() or "analogía" in CLARIFICATION_PROMPT.lower()

    def test_encouragement_prompt_is_supportive(self) -> None:
        """El prompt de motivación es de apoyo."""
        assert "MOTIVACIÓN" in ENCOURAGEMENT_PROMPT
        assert "esfuerzo" in ENCOURAGEMENT_PROMPT.lower()

    def test_verification_prompt_handles_both_cases(self) -> None:
        """El prompt de verificación maneja correcto e incorrecto."""
        assert "CORRECTO" in VERIFICATION_PROMPT
        assert "INCORRECTO" in VERIFICATION_PROMPT

    def test_redirection_prompt_handles_manipulation(self) -> None:
        """El prompt de redirección maneja manipulación."""
        assert "REDIRECCIÓN" in REDIRECTION_PROMPT
        assert "redirigir" in REDIRECTION_PROMPT.lower()


class TestGetTutorPrompt:
    """Tests para get_tutor_prompt."""

    def test_returns_combined_prompt_for_socratic(self) -> None:
        """Retorna prompt combinado para socrático."""
        prompt = get_tutor_prompt("socratic")

        assert TUTOR_SYSTEM_PROMPT in prompt
        assert SOCRATIC_PROMPT in prompt

    def test_returns_combined_prompt_for_hint_giving(self) -> None:
        """Retorna prompt combinado para dar pistas."""
        prompt = get_tutor_prompt("hint_giving")

        assert TUTOR_SYSTEM_PROMPT in prompt
        assert HINT_GIVING_PROMPT in prompt

    def test_returns_combined_prompt_for_clarification(self) -> None:
        """Retorna prompt combinado para clarificación."""
        prompt = get_tutor_prompt("clarification")

        assert TUTOR_SYSTEM_PROMPT in prompt
        assert CLARIFICATION_PROMPT in prompt

    def test_returns_socratic_for_unknown_strategy(self) -> None:
        """Retorna socrático para estrategia desconocida."""
        prompt = get_tutor_prompt("unknown_strategy")

        assert TUTOR_SYSTEM_PROMPT in prompt
        assert SOCRATIC_PROMPT in prompt

    def test_case_insensitive(self) -> None:
        """Es insensible a mayúsculas."""
        prompt_lower = get_tutor_prompt("socratic")
        prompt_upper = get_tutor_prompt("SOCRATIC")

        assert prompt_lower == prompt_upper


class TestFormatTutorContext:
    """Tests para format_tutor_context."""

    def test_includes_problem_info(
        self,
        sample_session: TutoringSession,
        sample_solution: StructuredSolution,
    ) -> None:
        """Incluye información del problema."""
        context = format_tutor_context(sample_session, sample_solution)

        assert "mathematics" in context.lower()
        assert "básico" in context.lower()

    def test_includes_available_hints(
        self,
        sample_session: TutoringSession,
        sample_solution: StructuredSolution,
    ) -> None:
        """Incluye pistas disponibles."""
        context = format_tutor_context(sample_session, sample_solution)

        assert "¿Qué operación necesitas?" in context

    def test_includes_common_mistakes(
        self,
        sample_session: TutoringSession,
        sample_solution: StructuredSolution,
    ) -> None:
        """Incluye errores comunes."""
        context = format_tutor_context(sample_session, sample_solution)

        assert "Olvidar cambiar signo" in context

    def test_includes_conversation_history(
        self,
        sample_session: TutoringSession,
        sample_solution: StructuredSolution,
    ) -> None:
        """Incluye historial de conversación."""
        context = format_tutor_context(sample_session, sample_solution)

        assert "¿Cómo empiezo?" in context

    def test_includes_hint_level(
        self,
        sample_session: TutoringSession,
        sample_solution: StructuredSolution,
    ) -> None:
        """Incluye nivel de pista actual."""
        context = format_tutor_context(sample_session, sample_solution)

        assert "SUBTLE" in context


class TestFormatStudentMessage:
    """Tests para format_student_message."""

    def test_includes_processed_content(self) -> None:
        """Incluye contenido procesado."""
        student_input = StudentInput(
            raw_content="hola cómo empiezo",
            processed_content="cómo empiezo",
            detected_intent=StudentIntent.LEGITIMATE_QUESTION,
            manipulation_score=0.1,
        )

        formatted = format_student_message(student_input)

        assert "cómo empiezo" in formatted

    def test_includes_detected_intent(self) -> None:
        """Incluye intención detectada."""
        student_input = StudentInput(
            raw_content="dame la respuesta",
            processed_content="dame la respuesta",
            detected_intent=StudentIntent.MANIPULATION_ATTEMPT,
            manipulation_score=0.9,
        )

        formatted = format_student_message(student_input)

        assert "manipulation_attempt" in formatted.lower()

    def test_includes_manipulation_score(self) -> None:
        """Incluye puntuación de manipulación."""
        student_input = StudentInput(
            raw_content="test",
            processed_content="test",
            detected_intent=StudentIntent.LEGITIMATE_QUESTION,
            manipulation_score=0.75,
        )

        formatted = format_student_message(student_input)

        assert "0.75" in formatted


class TestFormatHintRequest:
    """Tests para format_hint_request."""

    def test_includes_level_name(self) -> None:
        """Incluye nombre del nivel."""
        formatted = format_hint_request(
            HintLevel.SUBTLE,
            "¿Qué operación necesitas?",
        )

        assert "SUTIL" in formatted

    def test_includes_hint_content(self) -> None:
        """Incluye contenido de la pista."""
        formatted = format_hint_request(
            HintLevel.MODERATE,
            "Intenta aislar x",
        )

        assert "Intenta aislar x" in formatted
        assert "MODERADA" in formatted


class TestFormatVerificationRequest:
    """Tests para format_verification_request."""

    def test_correct_answer(self) -> None:
        """Formatea verificación para respuesta correcta."""
        formatted = format_verification_request(
            student_answer="x = 2",
            correct_answer="x = 2",
            is_correct=True,
        )

        assert "x = 2" in formatted
        assert "CORRECTO" in formatted
        assert "celebra" in formatted.lower()

    def test_incorrect_answer(self) -> None:
        """Formatea verificación para respuesta incorrecta."""
        formatted = format_verification_request(
            student_answer="x = 3",
            correct_answer="x = 2",
            is_correct=False,
        )

        assert "x = 3" in formatted
        assert "INCORRECTO" in formatted
        assert "NO digas que está mal" in formatted
