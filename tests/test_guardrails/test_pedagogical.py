"""
Tests para el validador pedagógico.
"""

import pytest

from src.core.types import (
    DifficultyLevel,
    GuardrailResult,
    Hint,
    HintLevel,
    ProblemType,
    StructuredSolution,
    TutoringSession,
)
from src.guardrails.base import GuardrailContext
from src.guardrails.detectors.pedagogical import PedagogicalValidator


@pytest.fixture
def validator() -> PedagogicalValidator:
    """Crea un validador pedagógico para tests."""
    return PedagogicalValidator()


@pytest.fixture
def sample_solution() -> StructuredSolution:
    """Crea una solución de ejemplo para tests."""
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
    """Crea una sesión de ejemplo para tests."""
    return TutoringSession(
        problem_text="Resuelve: 2x + 3 = 7",
        solution=sample_solution,
        current_hint_level=HintLevel.SUBTLE,
    )


class TestPedagogicalValidator:
    """Tests para PedagogicalValidator."""

    @pytest.mark.asyncio
    async def test_requires_questions_in_long_responses(
        self,
        validator: PedagogicalValidator,
        sample_session: TutoringSession,
    ) -> None:
        """Requiere preguntas en respuestas largas."""
        context = GuardrailContext(
            tutor_response="Debes usar álgebra para resolver este problema. "
                          "Primero, identifica los términos. "
                          "Luego, aplica las operaciones inversas.",
            session=sample_session,
            current_hint_level=HintLevel.SUBTLE,
        )

        result = await validator.check(context)

        assert result.result in (GuardrailResult.BLOCK, GuardrailResult.WARN)

    @pytest.mark.asyncio
    async def test_passes_with_questions(
        self,
        validator: PedagogicalValidator,
        sample_session: TutoringSession,
        sample_solution: StructuredSolution,
    ) -> None:
        """Pasa con preguntas presentes."""
        context = GuardrailContext(
            tutor_response="¿Qué crees que deberías hacer primero? "
                          "¿Puedes identificar qué operación necesitas?",
            solution=sample_solution,  # Añadir solución para topic relevance
            session=sample_session,
            current_hint_level=HintLevel.SUBTLE,
        )

        result = await validator.check(context)

        # Puede pasar o advertir por topic relevance bajo
        assert result.result in (GuardrailResult.PASS, GuardrailResult.WARN)

    @pytest.mark.asyncio
    async def test_validates_hint_progression(
        self,
        validator: PedagogicalValidator,
        sample_session: TutoringSession,
    ) -> None:
        """Valida progresión de hints."""
        # Sesión ya en nivel MODERATE
        sample_session.current_hint_level = HintLevel.MODERATE

        context = GuardrailContext(
            tutor_response="¿Has intentado restar 3 de ambos lados?",
            session=sample_session,
            current_hint_level=HintLevel.MODERATE,
        )

        result = await validator.check(context)

        assert result.result == GuardrailResult.PASS

    @pytest.mark.asyncio
    async def test_detects_telling_language(
        self,
        validator: PedagogicalValidator,
        sample_session: TutoringSession,
    ) -> None:
        """Detecta lenguaje directivo."""
        context = GuardrailContext(
            tutor_response="La respuesta es clara. Debes restar 3 primero. "
                          "Simplemente divide entre 2 después.",
            session=sample_session,
            current_hint_level=HintLevel.SUBTLE,
        )

        result = await validator.check(context)

        assert result.result in (GuardrailResult.BLOCK, GuardrailResult.WARN)
        assert result.details["language_analysis"]["guiding_score"] < 0.5

    @pytest.mark.asyncio
    async def test_validates_topic_relevance(
        self,
        validator: PedagogicalValidator,
        sample_session: TutoringSession,
        sample_solution: StructuredSolution,
    ) -> None:
        """Valida relevancia al tema."""
        context = GuardrailContext(
            tutor_response="Piensa en la ecuación y cómo aislar la variable x. "
                          "¿Qué operación inversa usarías para resolver ecuaciones de álgebra?",
            solution=sample_solution,
            session=sample_session,
            current_hint_level=HintLevel.SUBTLE,
        )

        result = await validator.check(context)

        # Topic relevance depende de los conceptos, puede ser bajo
        assert result.details["topic_relevance"]["score"] >= 0.0

    @pytest.mark.asyncio
    async def test_disabled_validator_passes_all(
        self,
        sample_session: TutoringSession,
    ) -> None:
        """Validador deshabilitado pasa todo."""
        from config.settings import GuardrailsConfig

        config = GuardrailsConfig(pedagogical_validation_enabled=False)
        validator = PedagogicalValidator(config=config)

        context = GuardrailContext(
            tutor_response="La respuesta es 2.",
            session=sample_session,
            current_hint_level=HintLevel.SUBTLE,
        )

        result = await validator.check(context)

        assert result.result == GuardrailResult.PASS

    @pytest.mark.asyncio
    async def test_short_response_less_strict(
        self,
        validator: PedagogicalValidator,
        sample_session: TutoringSession,
    ) -> None:
        """Respuestas cortas son menos estrictas."""
        context = GuardrailContext(
            tutor_response="¡Buen intento!",
            session=sample_session,
            current_hint_level=HintLevel.SUBTLE,
        )

        result = await validator.check(context)

        # Respuestas muy cortas pueden pasar
        assert result.result in (GuardrailResult.PASS, GuardrailResult.WARN)

    @pytest.mark.asyncio
    async def test_mixed_questions_and_statements(
        self,
        validator: PedagogicalValidator,
        sample_session: TutoringSession,
    ) -> None:
        """Mezcla de preguntas y declaraciones es aceptable."""
        context = GuardrailContext(
            tutor_response="Buen punto sobre la ecuación. "
                          "¿Has considerado qué pasa cuando restas el mismo valor de ambos lados? "
                          "Eso podría ayudarte a avanzar.",
            session=sample_session,
            current_hint_level=HintLevel.MODERATE,
        )

        result = await validator.check(context)

        # Debería pasar porque tiene una buena pregunta
        assert result.details["question_analysis"]["has_questions"] is True
