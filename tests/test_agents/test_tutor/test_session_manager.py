"""
Tests para el gestor de sesiones de tutoría.
"""

import pytest

from src.agents.tutor.session_manager import SessionManager
from src.core.types import (
    DifficultyLevel,
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
def session_manager() -> SessionManager:
    """Crea un gestor de sesiones para tests."""
    return SessionManager(max_sessions=100)


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
            SolutionStep(
                step_number=2,
                description="Dividir entre 2",
                reasoning="Para obtener x",
                result="x = 2",
                is_critical=True,
            ),
        ],
        final_answer="x = 2",
        hints=[
            Hint(level=HintLevel.SUBTLE, content="¿Qué operación necesitas?"),
            Hint(level=HintLevel.MODERATE, content="Intenta aislar x"),
            Hint(level=HintLevel.DIRECT, content="Resta 3 de ambos lados"),
        ],
    )


class TestSessionManager:
    """Tests para SessionManager."""

    @pytest.mark.asyncio
    async def test_create_session(
        self,
        session_manager: SessionManager,
        sample_solution: StructuredSolution,
    ) -> None:
        """Crea una sesión correctamente."""
        session = await session_manager.create_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        assert session is not None
        assert session.problem_text == "Resuelve: 2x + 3 = 7"
        assert session.solution == sample_solution
        assert session.current_hint_level == HintLevel.SUBTLE
        assert session.hints_given == 0
        assert session.questions_asked == 0
        assert session.student_reached_solution is False

    @pytest.mark.asyncio
    async def test_get_session(
        self,
        session_manager: SessionManager,
        sample_solution: StructuredSolution,
    ) -> None:
        """Obtiene una sesión existente."""
        session = await session_manager.create_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        retrieved = await session_manager.get_session(session.session_id)

        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(
        self,
        session_manager: SessionManager,
    ) -> None:
        """Retorna None para sesión inexistente."""
        from uuid import uuid4

        retrieved = await session_manager.get_session(uuid4())

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_update_session(
        self,
        session_manager: SessionManager,
        sample_solution: StructuredSolution,
    ) -> None:
        """Actualiza una sesión con un turno."""
        session = await session_manager.create_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        student_input = StudentInput(
            raw_content="cómo empiezo",
            processed_content="cómo empiezo",
            detected_intent=StudentIntent.LEGITIMATE_QUESTION,
        )

        tutor_response = TutorResponse(
            content="¿Qué operación crees que necesitas?",
            contains_question=True,
            hint_level_used=HintLevel.SUBTLE,
            strategy_used="socratic",
        )

        await session_manager.update_session(
            session_id=session.session_id,
            student_input=student_input,
            tutor_response=tutor_response,
        )

        updated = await session_manager.get_session(session.session_id)
        assert updated is not None
        assert updated.questions_asked == 1
        assert len(updated.conversation.messages) >= 2

    @pytest.mark.asyncio
    async def test_advance_hint_level(
        self,
        session_manager: SessionManager,
        sample_solution: StructuredSolution,
    ) -> None:
        """Avanza el nivel de pista."""
        session = await session_manager.create_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        # Nivel inicial: SUBTLE
        assert session.current_hint_level == HintLevel.SUBTLE

        # Avanzar a MODERATE
        new_level = await session_manager.advance_hint_level(session.session_id)
        assert new_level == HintLevel.MODERATE

        # Avanzar a DIRECT
        new_level = await session_manager.advance_hint_level(session.session_id)
        assert new_level == HintLevel.DIRECT

        # Mantenerse en DIRECT
        new_level = await session_manager.advance_hint_level(session.session_id)
        assert new_level == HintLevel.DIRECT

    @pytest.mark.asyncio
    async def test_check_student_solution_correct(
        self,
        session_manager: SessionManager,
        sample_solution: StructuredSolution,
    ) -> None:
        """Verifica respuesta correcta del estudiante."""
        session = await session_manager.create_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        is_correct = await session_manager.check_student_solution(
            session_id=session.session_id,
            student_answer="x = 2",
        )

        assert is_correct is True

        # Verificar que se marcó como resuelta
        updated = await session_manager.get_session(session.session_id)
        assert updated is not None
        assert updated.student_reached_solution is True
        assert updated.ended_at is not None

    @pytest.mark.asyncio
    async def test_check_student_solution_incorrect(
        self,
        session_manager: SessionManager,
        sample_solution: StructuredSolution,
    ) -> None:
        """Verifica respuesta incorrecta del estudiante."""
        session = await session_manager.create_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        is_correct = await session_manager.check_student_solution(
            session_id=session.session_id,
            student_answer="x = 3",
        )

        assert is_correct is False

        # Verificar que NO se marcó como resuelta
        updated = await session_manager.get_session(session.session_id)
        assert updated is not None
        assert updated.student_reached_solution is False

    @pytest.mark.asyncio
    async def test_check_student_solution_normalization(
        self,
        session_manager: SessionManager,
        sample_solution: StructuredSolution,
    ) -> None:
        """Normaliza respuestas para comparación."""
        session = await session_manager.create_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        # Probar diferentes formatos de la misma respuesta
        assert await session_manager.check_student_solution(
            session.session_id, "X = 2"
        ) is True

        # Crear nueva sesión para segunda prueba
        session2 = await session_manager.create_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        assert await session_manager.check_student_solution(
            session2.session_id, "x=2"
        ) is True

    @pytest.mark.asyncio
    async def test_end_session(
        self,
        session_manager: SessionManager,
        sample_solution: StructuredSolution,
    ) -> None:
        """Finaliza una sesión correctamente."""
        session = await session_manager.create_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        metrics = await session_manager.end_session(
            session_id=session.session_id,
            reason="user_requested",
        )

        assert metrics["session_id"] == str(session.session_id)
        assert metrics["end_reason"] == "user_requested"
        assert "duration_seconds" in metrics

        # Verificar que se eliminó de sesiones activas
        retrieved = await session_manager.get_session(session.session_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_session_metrics(
        self,
        session_manager: SessionManager,
        sample_solution: StructuredSolution,
    ) -> None:
        """Obtiene métricas de una sesión."""
        session = await session_manager.create_session(
            problem_text="Resuelve: 2x + 3 = 7",
            solution=sample_solution,
        )

        metrics = session_manager.get_session_metrics(session.session_id)

        assert metrics["session_id"] == str(session.session_id)
        assert metrics["problem_type"] == "mathematics"
        assert metrics["difficulty"] == "básico"
        assert metrics["questions_asked"] == 0
        assert metrics["hints_given"] == 0
        assert metrics["student_reached_solution"] is False

    @pytest.mark.asyncio
    async def test_session_limit(self) -> None:
        """Respeta límite de sesiones."""
        manager = SessionManager(max_sessions=2)

        solution = StructuredSolution(
            problem_text="Test",
            problem_type=ProblemType.MATHEMATICS,
            difficulty=DifficultyLevel.BASIC,
            concepts=[],
            steps=[],
            final_answer="x = 1",
        )

        # Crear 2 sesiones (límite)
        await manager.create_session("Test 1", solution)
        await manager.create_session("Test 2", solution)

        # Intentar crear tercera sesión debería fallar
        with pytest.raises(ValueError, match="Límite de sesiones"):
            await manager.create_session("Test 3", solution)

    def test_get_active_sessions_count(
        self,
        session_manager: SessionManager,
    ) -> None:
        """Cuenta sesiones activas."""
        assert session_manager.get_active_sessions_count() == 0

    @pytest.mark.asyncio
    async def test_get_all_session_ids(
        self,
        session_manager: SessionManager,
        sample_solution: StructuredSolution,
    ) -> None:
        """Lista IDs de sesiones activas."""
        session1 = await session_manager.create_session(
            "Test 1", sample_solution
        )
        session2 = await session_manager.create_session(
            "Test 2", sample_solution
        )

        ids = session_manager.get_all_session_ids()

        assert len(ids) == 2
        assert session1.session_id in ids
        assert session2.session_id in ids
