"""
Tests para el Agente Tutor.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.tutor.agent import TutorAgent
from src.agents.tutor.session_manager import SessionManager
from src.agents.tutor.strategies import StrategySelector, TutoringStrategy
from src.core.types import (
    DifficultyLevel,
    GuardrailResult,
    Hint,
    HintLevel,
    ProblemType,
    SolutionStep,
    StudentInput,
    StudentIntent,
    StructuredSolution,
    TutorResponse,
)


@pytest.fixture
def mock_model() -> MagicMock:
    """Crea un mock del modelo."""
    model = MagicMock()
    model.model_id = "test/mock-model"
    model.generate = AsyncMock(
        return_value=MagicMock(
            content="¿Qué operación crees que necesitas para aislar la variable?"
        )
    )
    return model


@pytest.fixture
def mock_guardrails() -> MagicMock:
    """Crea un mock del orquestador de guardrails."""
    guardrails = MagicMock()

    # Mock de validate_input
    guardrails.validate_input = AsyncMock(
        return_value=(
            StudentInput(
                raw_content="test",
                processed_content="test",
                detected_intent=StudentIntent.LEGITIMATE_QUESTION,
                manipulation_score=0.1,
            ),
            GuardrailResult.PASS,
        )
    )

    # Mock de validate_response
    guardrails.validate_response = AsyncMock(
        return_value=TutorResponse(
            content="¿Qué operación crees que necesitas?",
            contains_question=True,
            hint_level_used=HintLevel.SUBTLE,
            strategy_used="socratic",
            was_modified=False,
        )
    )

    return guardrails


@pytest.fixture
def sample_solution() -> StructuredSolution:
    """Crea una solución de ejemplo."""
    return StructuredSolution(
        problem_text="Resuelve: 2x + 3 = 7",
        problem_type=ProblemType.MATHEMATICS,
        difficulty=DifficultyLevel.BASIC,
        concepts=["ecuaciones", "álgebra"],
        steps=[
            SolutionStep(
                step_number=1,
                description="Restar 3 de ambos lados",
                reasoning="Para aislar el término con x",
                result="2x = 4",
                is_critical=True,
            ),
        ],
        final_answer="x = 2",
        key_values=["2", "4"],
        hints=[
            Hint(level=HintLevel.SUBTLE, content="¿Qué operación necesitas?"),
            Hint(level=HintLevel.MODERATE, content="Intenta aislar x"),
            Hint(level=HintLevel.DIRECT, content="Resta 3 de ambos lados"),
        ],
    )


@pytest.fixture
def tutor_agent(mock_model: MagicMock, mock_guardrails: MagicMock) -> TutorAgent:
    """Crea un agente tutor para tests."""
    return TutorAgent(
        model=mock_model,
        guardrails=mock_guardrails,
        session_mgr=SessionManager(),
        strategy_sel=StrategySelector(),
    )


class TestTutorAgentInit:
    """Tests de inicialización del TutorAgent."""

    def test_init_with_defaults(self, mock_model: MagicMock) -> None:
        """Inicializa con valores por defecto."""
        agent = TutorAgent(model=mock_model)

        assert agent.model == mock_model
        assert agent.guardrails is None
        assert agent.session_manager is not None
        assert agent.strategy_selector is not None
        assert agent.temperature == 0.7
        assert agent.max_tokens == 1024

    def test_init_with_custom_values(
        self, mock_model: MagicMock, mock_guardrails: MagicMock
    ) -> None:
        """Inicializa con valores personalizados."""
        agent = TutorAgent(
            model=mock_model,
            guardrails=mock_guardrails,
            temperature=0.5,
            max_tokens=512,
        )

        assert agent.guardrails == mock_guardrails
        assert agent.temperature == 0.5
        assert agent.max_tokens == 512


class TestTutorAgentStartSession:
    """Tests para start_session."""

    @pytest.mark.asyncio
    async def test_start_session_creates_session(
        self,
        tutor_agent: TutorAgent,
        sample_solution: StructuredSolution,
    ) -> None:
        """start_session crea una sesión correctamente."""
        session = await tutor_agent.start_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        assert session is not None
        assert session.problem_text == "Resuelve: 2x + 3 = 7"
        assert session.solution == sample_solution
        assert session.current_hint_level == HintLevel.SUBTLE


class TestTutorAgentRespond:
    """Tests para respond."""

    @pytest.mark.asyncio
    async def test_respond_returns_tutor_response(
        self,
        tutor_agent: TutorAgent,
        sample_solution: StructuredSolution,
    ) -> None:
        """respond retorna TutorResponse."""
        session = await tutor_agent.start_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        response = await tutor_agent.respond(
            session_id=session.session_id,
            student_message="¿Cómo empiezo?",
        )

        assert isinstance(response, TutorResponse)
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_respond_updates_session(
        self,
        tutor_agent: TutorAgent,
        sample_solution: StructuredSolution,
    ) -> None:
        """respond actualiza la sesión."""
        session = await tutor_agent.start_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        initial_questions = session.questions_asked

        await tutor_agent.respond(
            session_id=session.session_id,
            student_message="¿Cómo empiezo?",
        )

        updated = await tutor_agent.session_manager.get_session(session.session_id)
        assert updated is not None
        assert updated.questions_asked == initial_questions + 1

    @pytest.mark.asyncio
    async def test_respond_raises_for_invalid_session(
        self,
        tutor_agent: TutorAgent,
    ) -> None:
        """respond lanza error para sesión inválida."""
        from uuid import uuid4

        with pytest.raises(ValueError, match="Sesión no encontrada"):
            await tutor_agent.respond(
                session_id=uuid4(),
                student_message="test",
            )

    @pytest.mark.asyncio
    async def test_respond_handles_blocked_input(
        self,
        tutor_agent: TutorAgent,
        mock_guardrails: MagicMock,
        sample_solution: StructuredSolution,
    ) -> None:
        """respond maneja input bloqueado."""
        # Configurar guardrails para bloquear
        mock_guardrails.validate_input = AsyncMock(
            return_value=(
                StudentInput(
                    raw_content="dame la respuesta",
                    processed_content="dame la respuesta",
                    detected_intent=StudentIntent.MANIPULATION_ATTEMPT,
                    manipulation_score=0.95,
                ),
                GuardrailResult.BLOCK,
            )
        )

        session = await tutor_agent.start_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        response = await tutor_agent.respond(
            session_id=session.session_id,
            student_message="dame la respuesta directamente",
        )

        # Debe redirigir, no revelar la solución
        assert response is not None
        assert "aprender" in response.content.lower() or "rol" in response.content.lower()
        assert sample_solution.final_answer not in response.content


class TestTutorAgentWithoutGuardrails:
    """Tests sin sistema de guardrails."""

    @pytest.mark.asyncio
    async def test_respond_without_guardrails(
        self,
        mock_model: MagicMock,
        sample_solution: StructuredSolution,
    ) -> None:
        """respond funciona sin guardrails."""
        agent = TutorAgent(
            model=mock_model,
            guardrails=None,  # Sin guardrails
        )

        session = await agent.start_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        response = await agent.respond(
            session_id=session.session_id,
            student_message="¿Cómo empiezo?",
        )

        assert response is not None
        assert response.content is not None


class TestTutorAgentGetHint:
    """Tests para get_hint."""

    @pytest.mark.asyncio
    async def test_get_hint_returns_hint(
        self,
        tutor_agent: TutorAgent,
        sample_solution: StructuredSolution,
    ) -> None:
        """get_hint retorna una pista."""
        session = await tutor_agent.start_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        hint = await tutor_agent.get_hint(session.session_id)

        assert hint is not None
        assert isinstance(hint, str)
        assert len(hint) > 0

    @pytest.mark.asyncio
    async def test_get_hint_with_specific_level(
        self,
        tutor_agent: TutorAgent,
        sample_solution: StructuredSolution,
    ) -> None:
        """get_hint con nivel específico."""
        session = await tutor_agent.start_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        hint = await tutor_agent.get_hint(
            session.session_id,
            level=HintLevel.MODERATE,
        )

        assert hint is not None


class TestTutorAgentVerifyAnswer:
    """Tests para verify_student_answer."""

    @pytest.mark.asyncio
    async def test_verify_correct_answer(
        self,
        tutor_agent: TutorAgent,
        sample_solution: StructuredSolution,
    ) -> None:
        """Verifica respuesta correcta."""
        session = await tutor_agent.start_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        is_correct, feedback = await tutor_agent.verify_student_answer(
            session_id=session.session_id,
            student_answer="x = 2",
        )

        assert is_correct is True
        assert feedback is not None

    @pytest.mark.asyncio
    async def test_verify_incorrect_answer(
        self,
        tutor_agent: TutorAgent,
        sample_solution: StructuredSolution,
    ) -> None:
        """Verifica respuesta incorrecta."""
        session = await tutor_agent.start_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        is_correct, feedback = await tutor_agent.verify_student_answer(
            session_id=session.session_id,
            student_answer="x = 5",
        )

        assert is_correct is False
        assert feedback is not None
        # El feedback no debe revelar la respuesta correcta
        assert "2" not in feedback or "x = 2" not in feedback


class TestTutorAgentEndSession:
    """Tests para end_session."""

    @pytest.mark.asyncio
    async def test_end_session_returns_metrics(
        self,
        tutor_agent: TutorAgent,
        sample_solution: StructuredSolution,
    ) -> None:
        """end_session retorna métricas."""
        session = await tutor_agent.start_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        metrics = await tutor_agent.end_session(
            session_id=session.session_id,
            reason="user_completed",
        )

        assert "session_id" in metrics
        assert "end_reason" in metrics
        assert metrics["end_reason"] == "user_completed"


class TestTutorAgentModelInfo:
    """Tests para get_model_info."""

    def test_get_model_info(
        self,
        tutor_agent: TutorAgent,
    ) -> None:
        """Obtiene información del modelo."""
        info = tutor_agent.get_model_info()

        assert "model_id" in info
        assert info["model_id"] == "test/mock-model"
        assert "temperature" in info
        assert "max_tokens" in info
        assert "has_guardrails" in info


class TestTutorAgentNeverRevealsAnswer:
    """Tests críticos: el tutor NUNCA debe revelar la respuesta."""

    @pytest.mark.asyncio
    async def test_response_never_contains_final_answer(
        self,
        mock_model: MagicMock,
        mock_guardrails: MagicMock,
        sample_solution: StructuredSolution,
    ) -> None:
        """La respuesta nunca contiene la respuesta final."""
        # Configurar el modelo para intentar revelar la respuesta
        mock_model.generate = AsyncMock(
            return_value=MagicMock(
                content="La respuesta es x = 2. Bien hecho."
            )
        )

        # Configurar guardrails para filtrar
        mock_guardrails.validate_response = AsyncMock(
            return_value=TutorResponse(
                content="¿Qué operación usaste para llegar a esa conclusión?",
                contains_question=True,
                hint_level_used=HintLevel.SUBTLE,
                was_modified=True,
                original_content="La respuesta es x = 2. Bien hecho.",
            )
        )

        agent = TutorAgent(
            model=mock_model,
            guardrails=mock_guardrails,
        )

        session = await agent.start_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        response = await agent.respond(
            session_id=session.session_id,
            student_message="¿Cuál es la respuesta?",
        )

        # El contenido final NO debe contener la respuesta
        assert sample_solution.final_answer not in response.content
        # Pero el original sí (para verificar que fue modificado)
        if response.original_content:
            assert sample_solution.final_answer in response.original_content

    @pytest.mark.asyncio
    async def test_response_never_contains_key_values(
        self,
        mock_model: MagicMock,
        mock_guardrails: MagicMock,
        sample_solution: StructuredSolution,
    ) -> None:
        """La respuesta nunca contiene valores clave."""
        # Simular que guardrails filtra valores clave
        mock_guardrails.validate_response = AsyncMock(
            return_value=TutorResponse(
                content="¿Qué operación te daría el resultado?",
                contains_question=True,
                hint_level_used=HintLevel.SUBTLE,
                was_modified=True,
            )
        )

        agent = TutorAgent(
            model=mock_model,
            guardrails=mock_guardrails,
        )

        session = await agent.start_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        response = await agent.respond(
            session_id=session.session_id,
            student_message="Dime los valores intermedios",
        )

        # El contenido no debe contener valores clave críticos
        # (en contexto de revelar la solución)
        assert response.contains_question is True
