"""
Tests para el Agente Solucionador.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.solver.agent import SolutionCache, SolverAgent
from src.core.types import (
    DifficultyLevel,
    HintLevel,
    ModelResponse,
    ProblemType,
    StructuredSolution,
)


class TestSolutionCache:
    """Tests para SolutionCache."""
    
    @pytest.fixture
    def cache(self) -> SolutionCache:
        return SolutionCache(max_size=10, ttl_seconds=60)
    
    @pytest.fixture
    def sample_solution(self) -> StructuredSolution:
        from src.core.types import Hint, SolutionStep
        
        return StructuredSolution(
            problem_text="2x + 3 = 7",
            problem_type=ProblemType.MATHEMATICS,
            difficulty=DifficultyLevel.BASIC,
            steps=[SolutionStep(
                step_number=1,
                description="Resolver",
                reasoning="Despeje",
            )],
            final_answer="x = 2",
            hints=[Hint(level=HintLevel.SUBTLE, content="Piensa")],
        )
    
    def test_cache_miss(self, cache: SolutionCache) -> None:
        """Test de cache miss."""
        result = cache.get("problema no cacheado")
        assert result is None
    
    def test_cache_hit(
        self,
        cache: SolutionCache,
        sample_solution: StructuredSolution,
    ) -> None:
        """Test de cache hit."""
        cache.set("mi problema", sample_solution)
        
        result = cache.get("mi problema")
        
        assert result is not None
        assert result.final_answer == "x = 2"
    
    def test_cache_key_normalization(
        self,
        cache: SolutionCache,
        sample_solution: StructuredSolution,
    ) -> None:
        """Test de que las claves se normalizan."""
        cache.set("Mi  Problema  ", sample_solution)
        
        # Debería encontrarlo con espaciado diferente
        result = cache.get("mi problema")
        assert result is not None
    
    def test_cache_max_size(self, sample_solution: StructuredSolution) -> None:
        """Test de límite de tamaño."""
        cache = SolutionCache(max_size=3, ttl_seconds=60)
        
        cache.set("problema 1", sample_solution)
        cache.set("problema 2", sample_solution)
        cache.set("problema 3", sample_solution)
        cache.set("problema 4", sample_solution)
        
        # No debería exceder max_size
        assert len(cache) <= 3
    
    def test_cache_clear(
        self,
        cache: SolutionCache,
        sample_solution: StructuredSolution,
    ) -> None:
        """Test de limpieza de caché."""
        cache.set("problema", sample_solution)
        assert len(cache) == 1
        
        cache.clear()
        
        assert len(cache) == 0


class TestSolverAgent:
    """Tests para SolverAgent."""
    
    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Crea un mock del modelo."""
        model = MagicMock()
        model.model_id = "ollama/test-model"
        model.generate = AsyncMock()
        return model
    
    @pytest.fixture
    def solver(self, mock_model: MagicMock) -> SolverAgent:
        """Crea un SolverAgent con modelo mock."""
        return SolverAgent(
            model=mock_model,
            use_cache=True,
            max_retries=2,
        )
    
    @pytest.fixture
    def valid_model_response(self) -> str:
        """Respuesta válida del modelo."""
        return '''{
  "problem_type": "mathematics",
  "difficulty": "básico",
  "concepts": ["ecuaciones lineales"],
  "prerequisites": ["aritmética"],
  "solution": {
    "steps": [
      {
        "step_number": 1,
        "description": "Identificar ecuación",
        "reasoning": "2x + 3 = 7",
        "is_critical": false
      },
      {
        "step_number": 2,
        "description": "Despejar x",
        "reasoning": "Restar 3, dividir entre 2",
        "calculation": "x = (7-3)/2 = 2",
        "result": "x = 2",
        "is_critical": true
      }
    ]
  },
  "final_answer": "x = 2",
  "verification": "2(2) + 3 = 7 ✓",
  "common_mistakes": ["Olvidar cambiar signo"],
  "hints": [
    {"level": 1, "content": "¿Qué operación necesitas?", "concepts_referenced": ["despeje"]},
    {"level": 2, "content": "Aísla la variable", "concepts_referenced": ["ecuaciones"]},
    {"level": 3, "content": "Resta 3 y divide entre 2", "concepts_referenced": ["operaciones"]}
  ],
  "theory_references": ["Propiedad de igualdad"],
  "key_values": ["2", "7", "3"]
}'''
    
    @pytest.mark.asyncio
    async def test_solve_success(
        self,
        solver: SolverAgent,
        mock_model: MagicMock,
        valid_model_response: str,
    ) -> None:
        """Test de resolución exitosa."""
        mock_model.generate.return_value = ModelResponse(
            content=valid_model_response,
            model="test",
        )
        
        solution = await solver.solve("Resuelve: 2x + 3 = 7")
        
        assert solution.final_answer == "x = 2"
        assert solution.problem_type == ProblemType.MATHEMATICS
        assert len(solution.steps) == 2
        assert len(solution.hints) >= 3
        assert solution.solver_model == "ollama/test-model"
    
    @pytest.mark.asyncio
    async def test_solve_with_domain_hint(
        self,
        solver: SolverAgent,
        mock_model: MagicMock,
        valid_model_response: str,
    ) -> None:
        """Test de resolución con hint de dominio."""
        mock_model.generate.return_value = ModelResponse(
            content=valid_model_response,
            model="test",
        )
        
        await solver.solve(
            "Resuelve: 2x + 3 = 7",
            domain_hint="mathematics",
        )
        
        # Verificar que se llamó al modelo
        mock_model.generate.assert_called_once()
        
        # Verificar que el prompt contiene info de matemáticas
        call_args = mock_model.generate.call_args
        messages = call_args[0][0]
        system_prompt = messages[0]["content"]
        assert "matemátic" in system_prompt.lower() or "algebra" in system_prompt.lower()
    
    @pytest.mark.asyncio
    async def test_solve_uses_cache(
        self,
        solver: SolverAgent,
        mock_model: MagicMock,
        valid_model_response: str,
    ) -> None:
        """Test de que usa caché correctamente."""
        mock_model.generate.return_value = ModelResponse(
            content=valid_model_response,
            model="test",
        )
        
        # Primera llamada
        await solver.solve("Problema de prueba")
        
        # Segunda llamada con mismo problema
        await solver.solve("Problema de prueba")
        
        # Solo debería haber llamado al modelo una vez
        assert mock_model.generate.call_count == 1
    
    @pytest.mark.asyncio
    async def test_solve_force_regenerate(
        self,
        solver: SolverAgent,
        mock_model: MagicMock,
        valid_model_response: str,
    ) -> None:
        """Test de forzar regeneración ignorando caché."""
        mock_model.generate.return_value = ModelResponse(
            content=valid_model_response,
            model="test",
        )
        
        await solver.solve("Problema")
        await solver.solve("Problema", force_regenerate=True)
        
        # Debería haber llamado dos veces
        assert mock_model.generate.call_count == 2
    
    @pytest.mark.asyncio
    async def test_solve_retries_on_parse_error(
        self,
        solver: SolverAgent,
        mock_model: MagicMock,
        valid_model_response: str,
    ) -> None:
        """Test de reintentos cuando falla el parsing."""
        # Primera llamada devuelve JSON inválido
        mock_model.generate.side_effect = [
            ModelResponse(content='{"invalid": json', model="test"),
            ModelResponse(content=valid_model_response, model="test"),
        ]
        
        solution = await solver.solve("Problema")
        
        # Debería haber reintentado
        assert mock_model.generate.call_count == 2
        assert solution.final_answer == "x = 2"
    
    @pytest.mark.asyncio
    async def test_solve_with_verification_match(
        self,
        solver: SolverAgent,
        mock_model: MagicMock,
        valid_model_response: str,
    ) -> None:
        """Test de verificación cuando coincide."""
        mock_model.generate.return_value = ModelResponse(
            content=valid_model_response,
            model="test",
        )
        
        solution, matches = await solver.solve_with_verification(
            "2x + 3 = 7",
            expected_answer="x = 2",
        )
        
        assert matches is True
    
    @pytest.mark.asyncio
    async def test_solve_with_verification_mismatch(
        self,
        solver: SolverAgent,
        mock_model: MagicMock,
        valid_model_response: str,
    ) -> None:
        """Test de verificación cuando no coincide."""
        mock_model.generate.return_value = ModelResponse(
            content=valid_model_response,
            model="test",
        )
        
        solution, matches = await solver.solve_with_verification(
            "2x + 3 = 7",
            expected_answer="x = 5",  # Incorrecto
        )
        
        assert matches is False
    
    @pytest.mark.asyncio
    async def test_get_hints_for_level(
        self,
        solver: SolverAgent,
        mock_model: MagicMock,
        valid_model_response: str,
    ) -> None:
        """Test de obtención de pistas por nivel."""
        mock_model.generate.return_value = ModelResponse(
            content=valid_model_response,
            model="test",
        )
        
        solution = await solver.solve("Problema")
        
        # Nivel 1 solo
        hints_1 = await solver.get_hints_for_level(solution, HintLevel.SUBTLE)
        assert len(hints_1) >= 1
        
        # Hasta nivel 3
        hints_3 = await solver.get_hints_for_level(solution, HintLevel.DIRECT)
        assert len(hints_3) >= len(hints_1)
    
    def test_get_cache_stats(self, solver: SolverAgent) -> None:
        """Test de estadísticas de caché."""
        stats = solver.get_cache_stats()
        
        assert stats["enabled"] is True
        assert "size" in stats
        assert "max_size" in stats
    
    def test_get_cache_stats_disabled(self, mock_model: MagicMock) -> None:
        """Test de estadísticas con caché deshabilitada."""
        solver = SolverAgent(model=mock_model, use_cache=False)
        
        stats = solver.get_cache_stats()
        
        assert stats["enabled"] is False
    
    def test_clear_cache(
        self,
        solver: SolverAgent,
    ) -> None:
        """Test de limpieza de caché."""
        # Añadir algo a la caché manualmente
        from src.core.types import Hint, SolutionStep
        
        solution = StructuredSolution(
            problem_text="test",
            problem_type=ProblemType.MATHEMATICS,
            difficulty=DifficultyLevel.BASIC,
            steps=[SolutionStep(step_number=1, description="", reasoning="")],
            final_answer="42",
            hints=[Hint(level=HintLevel.SUBTLE, content="")],
        )
        solver.cache.set("test", solution)
        
        assert len(solver.cache) == 1
        
        solver.clear_cache()
        
        assert len(solver.cache) == 0
    
    def test_normalize_answer(self, solver: SolverAgent) -> None:
        """Test de normalización de respuestas."""
        # Espacios
        assert solver._normalize_answer("x = 2") == "x=2"
        
        # Mayúsculas
        assert solver._normalize_answer("X = 2") == "x=2"
        
        # Puntuación al final
        assert solver._normalize_answer("x = 2.") == "x=2"


class TestSolverAgentCreate:
    """Tests para el factory method create()."""
    
    @pytest.mark.asyncio
    async def test_create_with_default_model(self, mock_settings: MagicMock) -> None:
        """Test de creación con modelo por defecto."""
        with patch("src.agents.solver.agent.get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.model_id = "ollama/qwen2.5:14b"
            mock_model.health_check = AsyncMock(return_value=True)
            mock_get_model.return_value = mock_model
            
            solver = await SolverAgent.create()
            
            assert solver.model.model_id == "ollama/qwen2.5:14b"
    
    @pytest.mark.asyncio
    async def test_create_with_custom_model(self, mock_settings: MagicMock) -> None:
        """Test de creación con modelo personalizado."""
        with patch("src.agents.solver.agent.get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.model_id = "ollama/llama3.1:70b"
            mock_model.health_check = AsyncMock(return_value=True)
            mock_get_model.return_value = mock_model
            
            solver = await SolverAgent.create("ollama/llama3.1:70b")
            
            mock_get_model.assert_called_with("ollama/llama3.1:70b")


# Tests de integración (requieren Ollama)
@pytest.mark.integration
class TestSolverIntegration:
    """Tests de integración con modelo real."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_solve_real_problem(self, skip_if_no_ollama: None) -> None:
        """Test con problema real usando Ollama."""
        solver = await SolverAgent.create("ollama/llama3.2:1b")
        
        solution = await solver.solve(
            "¿Cuánto es 2 + 2?",
            domain_hint="mathematics",
        )
        
        # Verificar estructura básica
        assert solution.final_answer is not None
        assert len(solution.steps) >= 1
        assert solution.problem_type == ProblemType.MATHEMATICS
