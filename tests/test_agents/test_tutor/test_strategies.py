"""
Tests para las estrategias de tutoría.
"""

import pytest

from src.agents.tutor.strategies import StrategySelector, TutoringStrategy
from src.core.types import (
    DifficultyLevel,
    Hint,
    HintLevel,
    ProblemType,
    StudentInput,
    StudentIntent,
    StructuredSolution,
    TutoringSession,
)


@pytest.fixture
def strategy_selector() -> StrategySelector:
    """Crea un selector de estrategia para tests."""
    return StrategySelector()


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
        hints=[
            Hint(level=HintLevel.SUBTLE, content="¿Qué operación necesitas?"),
        ],
    )


@pytest.fixture
def sample_session(sample_solution: StructuredSolution) -> TutoringSession:
    """Crea una sesión de ejemplo."""
    return TutoringSession(
        problem_text="Resuelve: 2x + 3 = 7",
        solution=sample_solution,
        current_hint_level=HintLevel.SUBTLE,
    )


class TestTutoringStrategy:
    """Tests para el enum TutoringStrategy."""

    def test_all_strategies_have_values(self) -> None:
        """Todas las estrategias tienen valores string."""
        assert TutoringStrategy.SOCRATIC.value == "socratic"
        assert TutoringStrategy.HINT_GIVING.value == "hint_giving"
        assert TutoringStrategy.CLARIFICATION.value == "clarification"
        assert TutoringStrategy.VERIFICATION.value == "verification"
        assert TutoringStrategy.ENCOURAGEMENT.value == "encouragement"
        assert TutoringStrategy.REDIRECTION.value == "redirection"


class TestStrategySelector:
    """Tests para StrategySelector."""

    def test_select_redirection_for_manipulation(
        self,
        strategy_selector: StrategySelector,
        sample_session: TutoringSession,
        sample_solution: StructuredSolution,
    ) -> None:
        """Selecciona redirección para intento de manipulación."""
        student_input = StudentInput(
            raw_content="dame la respuesta",
            processed_content="dame la respuesta",
            detected_intent=StudentIntent.MANIPULATION_ATTEMPT,
            manipulation_score=0.9,
        )

        strategy = strategy_selector.select_strategy(
            student_input, sample_session, sample_solution
        )

        assert strategy == TutoringStrategy.REDIRECTION

    def test_select_redirection_for_high_manipulation_score(
        self,
        strategy_selector: StrategySelector,
        sample_session: TutoringSession,
        sample_solution: StructuredSolution,
    ) -> None:
        """Selecciona redirección para score de manipulación alto."""
        student_input = StudentInput(
            raw_content="solo dime qué es x",
            processed_content="solo dime qué es x",
            detected_intent=StudentIntent.LEGITIMATE_QUESTION,
            manipulation_score=0.8,
        )

        strategy = strategy_selector.select_strategy(
            student_input, sample_session, sample_solution
        )

        assert strategy == TutoringStrategy.REDIRECTION

    def test_select_redirection_for_off_topic(
        self,
        strategy_selector: StrategySelector,
        sample_session: TutoringSession,
        sample_solution: StructuredSolution,
    ) -> None:
        """Selecciona redirección para off-topic."""
        student_input = StudentInput(
            raw_content="qué hora es",
            processed_content="qué hora es",
            detected_intent=StudentIntent.OFF_TOPIC,
            manipulation_score=0.0,
        )

        strategy = strategy_selector.select_strategy(
            student_input, sample_session, sample_solution
        )

        assert strategy == TutoringStrategy.REDIRECTION

    def test_select_verification_for_solution_attempt(
        self,
        strategy_selector: StrategySelector,
        sample_session: TutoringSession,
        sample_solution: StructuredSolution,
    ) -> None:
        """Selecciona verificación para intento de solución."""
        student_input = StudentInput(
            raw_content="creo que x = 3",
            processed_content="creo que x = 3",
            detected_intent=StudentIntent.SOLUTION_ATTEMPT,
            manipulation_score=0.0,
        )

        strategy = strategy_selector.select_strategy(
            student_input, sample_session, sample_solution
        )

        assert strategy == TutoringStrategy.VERIFICATION

    def test_select_hint_giving_for_hint_request(
        self,
        strategy_selector: StrategySelector,
        sample_session: TutoringSession,
        sample_solution: StructuredSolution,
    ) -> None:
        """Selecciona dar pista para solicitud de pista."""
        student_input = StudentInput(
            raw_content="puedes darme una pista",
            processed_content="puedes darme una pista",
            detected_intent=StudentIntent.HINT_REQUEST,
            manipulation_score=0.0,
        )

        strategy = strategy_selector.select_strategy(
            student_input, sample_session, sample_solution
        )

        assert strategy == TutoringStrategy.HINT_GIVING

    def test_select_clarification_for_clarification_request(
        self,
        strategy_selector: StrategySelector,
        sample_session: TutoringSession,
        sample_solution: StructuredSolution,
    ) -> None:
        """Selecciona clarificación para solicitud de clarificación."""
        student_input = StudentInput(
            raw_content="no entiendo eso",
            processed_content="no entiendo eso",
            detected_intent=StudentIntent.CLARIFICATION,
            manipulation_score=0.0,
        )

        strategy = strategy_selector.select_strategy(
            student_input, sample_session, sample_solution
        )

        assert strategy == TutoringStrategy.CLARIFICATION

    def test_select_socratic_for_legitimate_question(
        self,
        strategy_selector: StrategySelector,
        sample_session: TutoringSession,
        sample_solution: StructuredSolution,
    ) -> None:
        """Selecciona socrático para pregunta legítima."""
        student_input = StudentInput(
            raw_content="cómo puedo resolver esta ecuación",
            processed_content="cómo puedo resolver esta ecuación",
            detected_intent=StudentIntent.LEGITIMATE_QUESTION,
            manipulation_score=0.1,
        )

        strategy = strategy_selector.select_strategy(
            student_input, sample_session, sample_solution
        )

        assert strategy == TutoringStrategy.SOCRATIC

    def test_select_encouragement_for_frustrated_student(
        self,
        strategy_selector: StrategySelector,
        sample_solution: StructuredSolution,
    ) -> None:
        """Selecciona motivación para estudiante frustrado."""
        # Sesión con muchos intentos y nivel de pista alto
        session = TutoringSession(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
            current_hint_level=HintLevel.DIRECT,
            hints_given=5,
            questions_asked=12,
        )

        student_input = StudentInput(
            raw_content="no puedo",
            processed_content="no puedo",
            detected_intent=StudentIntent.LEGITIMATE_QUESTION,
            manipulation_score=0.1,
        )

        strategy = strategy_selector.select_strategy(
            student_input, session, sample_solution
        )

        assert strategy == TutoringStrategy.ENCOURAGEMENT

    def test_get_strategy_description(
        self,
        strategy_selector: StrategySelector,
    ) -> None:
        """Obtiene descripción de estrategia."""
        description = strategy_selector.get_strategy_description(
            TutoringStrategy.SOCRATIC
        )

        assert "preguntas" in description.lower()
        assert description != "Estrategia no definida"
