"""
Tests para el detector de fugas de solución.
"""

import pytest

from src.core.types import (
    DifficultyLevel,
    GuardrailResult,
    Hint,
    HintLevel,
    ProblemType,
    SolutionStep,
    StructuredSolution,
)
from src.guardrails.base import GuardrailContext
from src.guardrails.detectors.solution_leak import SolutionLeakDetector


@pytest.fixture
def detector() -> SolutionLeakDetector:
    """Crea un detector de fugas para tests."""
    return SolutionLeakDetector()


@pytest.fixture
def sample_solution() -> StructuredSolution:
    """Crea una solución de ejemplo para tests."""
    return StructuredSolution(
        problem_text="Resuelve: 2x + 3 = 7",
        problem_type=ProblemType.MATHEMATICS,
        difficulty=DifficultyLevel.BASIC,
        concepts=["ecuaciones lineales", "álgebra"],
        steps=[
            SolutionStep(
                step_number=1,
                description="Restar 3 de ambos lados",
                reasoning="Para aislar el término con x",
                calculation="2x = 7 - 3",
                result="2x = 4",
                is_critical=True,
            ),
            SolutionStep(
                step_number=2,
                description="Dividir ambos lados entre 2",
                reasoning="Para obtener el valor de x",
                calculation="x = 4 / 2",
                result="x = 2",
                is_critical=True,
            ),
        ],
        final_answer="x = 2",
        verification="2(2) + 3 = 4 + 3 = 7 ✓",
        key_values=["2", "4", "7"],
        hints=[
            Hint(level=HintLevel.SUBTLE, content="¿Qué operación necesitas primero?"),
            Hint(level=HintLevel.MODERATE, content="Intenta aislar x"),
            Hint(level=HintLevel.DIRECT, content="Resta el 3 de ambos lados"),
        ],
    )


class TestSolutionLeakDetector:
    """Tests para SolutionLeakDetector."""

    @pytest.mark.asyncio
    async def test_detects_key_value_leak(
        self,
        detector: SolutionLeakDetector,
        sample_solution: StructuredSolution,
    ) -> None:
        """Detecta valores clave filtrados."""
        context = GuardrailContext(
            tutor_response="El valor que buscas es 2",
            solution=sample_solution,
        )

        with pytest.raises(Exception):  # SolutionLeakDetectedError
            await detector.check(context)

    @pytest.mark.asyncio
    async def test_detects_final_answer_leak(
        self,
        detector: SolutionLeakDetector,
        sample_solution: StructuredSolution,
    ) -> None:
        """Detecta respuesta final filtrada."""
        context = GuardrailContext(
            tutor_response="La respuesta es x = 2",
            solution=sample_solution,
        )

        with pytest.raises(Exception):  # SolutionLeakDetectedError
            await detector.check(context)

    @pytest.mark.asyncio
    async def test_detects_critical_step_leak(
        self,
        detector: SolutionLeakDetector,
        sample_solution: StructuredSolution,
    ) -> None:
        """Detecta pasos críticos filtrados (lanza excepción por key_value)."""
        from src.core.exceptions import SolutionLeakDetectedError

        context = GuardrailContext(
            tutor_response="El primer paso es hacer 2x = 4",
            solution=sample_solution,
        )

        # Contiene "4" que es un key_value, lanza excepción
        with pytest.raises(SolutionLeakDetectedError):
            await detector.check(context)

    @pytest.mark.asyncio
    async def test_allows_safe_hints(
        self,
        detector: SolutionLeakDetector,
        sample_solution: StructuredSolution,
    ) -> None:
        """Permite pistas seguras."""
        context = GuardrailContext(
            tutor_response="¿Qué operación usarías para eliminar el 3 de la ecuación?",
            solution=sample_solution,
        )

        result = await detector.check(context)

        assert result.result == GuardrailResult.PASS

    @pytest.mark.asyncio
    async def test_allows_pedagogical_guidance(
        self,
        detector: SolutionLeakDetector,
        sample_solution: StructuredSolution,
    ) -> None:
        """Permite guía pedagógica sin fugas."""
        context = GuardrailContext(
            tutor_response="Piensa en qué operación es la inversa de la suma. "
                          "¿Cómo podrías aislar la variable?",
            solution=sample_solution,
        )

        result = await detector.check(context)

        assert result.result == GuardrailResult.PASS

    @pytest.mark.asyncio
    async def test_no_solution_context_passes(
        self,
        detector: SolutionLeakDetector,
    ) -> None:
        """Sin solución para comparar, pasa."""
        context = GuardrailContext(
            tutor_response="La respuesta es 42",
            solution=None,
        )

        result = await detector.check(context)

        assert result.result == GuardrailResult.PASS

    @pytest.mark.asyncio
    async def test_disabled_detector_passes_all(
        self,
        sample_solution: StructuredSolution,
    ) -> None:
        """Detector deshabilitado pasa todo."""
        from config.settings import GuardrailsConfig

        config = GuardrailsConfig(solution_leak_detection_enabled=False)
        detector = SolutionLeakDetector(config=config)

        context = GuardrailContext(
            tutor_response="La respuesta es x = 2",
            solution=sample_solution,
        )

        result = await detector.check(context)

        assert result.result == GuardrailResult.PASS

    @pytest.mark.asyncio
    async def test_detects_calculation_exposure(
        self,
        detector: SolutionLeakDetector,
        sample_solution: StructuredSolution,
    ) -> None:
        """Detecta exposición de cálculos (lanza excepción por key_value)."""
        from src.core.exceptions import SolutionLeakDetectedError

        context = GuardrailContext(
            tutor_response="Deberías calcular x = 4 / 2 para obtener el resultado",
            solution=sample_solution,
        )

        # Contiene "4" y "2" que son key_values, lanza excepción
        with pytest.raises(SolutionLeakDetectedError):
            await detector.check(context)

    @pytest.mark.asyncio
    async def test_empty_response_passes(
        self,
        detector: SolutionLeakDetector,
        sample_solution: StructuredSolution,
    ) -> None:
        """Respuesta vacía pasa."""
        context = GuardrailContext(
            tutor_response="",
            solution=sample_solution,
        )

        result = await detector.check(context)

        assert result.result == GuardrailResult.PASS
