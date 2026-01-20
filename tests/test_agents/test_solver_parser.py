"""
Tests para el parser de soluciones del Solver.
"""

from __future__ import annotations

import pytest

from src.agents.solver.parser import SolutionParser, SolutionRepairStrategy
from src.core.exceptions import JSONParsingError, SolutionParsingError
from src.core.types import DifficultyLevel, HintLevel, ProblemType


class TestSolutionParser:
    """Tests para SolutionParser."""
    
    @pytest.fixture
    def parser(self) -> SolutionParser:
        return SolutionParser()
    
    @pytest.fixture
    def valid_json_response(self) -> str:
        return '''{
  "problem_type": "mathematics",
  "difficulty": "básico",
  "concepts": ["ecuaciones lineales", "despeje"],
  "prerequisites": ["operaciones básicas"],
  "solution": {
    "steps": [
      {
        "step_number": 1,
        "description": "Identificar la ecuación",
        "reasoning": "Tenemos 2x + 3 = 7",
        "calculation": "2x + 3 = 7",
        "result": "Ecuación identificada",
        "is_critical": false
      },
      {
        "step_number": 2,
        "description": "Restar 3 de ambos lados",
        "reasoning": "Para aislar x",
        "calculation": "2x = 4",
        "result": "2x = 4",
        "is_critical": true
      },
      {
        "step_number": 3,
        "description": "Dividir entre 2",
        "reasoning": "Obtener x",
        "calculation": "x = 2",
        "result": "x = 2",
        "is_critical": true
      }
    ]
  },
  "final_answer": "x = 2",
  "verification": "2(2) + 3 = 7 ✓",
  "common_mistakes": ["Error de signo"],
  "hints": [
    {"level": 1, "content": "¿Qué operación usarías?", "concepts_referenced": ["despeje"]},
    {"level": 2, "content": "Primero elimina el 3", "concepts_referenced": ["suma"]},
    {"level": 3, "content": "Resta 3 y divide entre 2", "concepts_referenced": ["ecuaciones"]}
  ],
  "theory_references": ["Propiedad de la igualdad"],
  "key_values": ["2", "4", "7"]
}'''
    
    def test_parse_valid_json(
        self,
        parser: SolutionParser,
        valid_json_response: str,
    ) -> None:
        """Test de parsing de JSON válido."""
        solution = parser.parse(valid_json_response, "2x + 3 = 7")
        
        assert solution.problem_type == ProblemType.MATHEMATICS
        assert solution.difficulty == DifficultyLevel.BASIC
        assert len(solution.steps) == 3
        assert solution.final_answer == "x = 2"
        assert len(solution.hints) >= 3
        assert "2" in solution.key_values
    
    def test_parse_json_in_markdown_block(self, parser: SolutionParser) -> None:
        """Test de parsing con JSON en bloque markdown."""
        response = '''Aquí está la solución:

```json
{
  "problem_type": "mathematics",
  "difficulty": "básico",
  "concepts": [],
  "solution": {
    "steps": [
      {"step_number": 1, "description": "Paso 1", "reasoning": "Razón"}
    ]
  },
  "final_answer": "42",
  "hints": []
}
```

Espero que te sea útil.'''
        
        solution = parser.parse(response, "test problem")
        
        assert solution.final_answer == "42"
        assert len(solution.steps) == 1
    
    def test_parse_json_with_text_around(self, parser: SolutionParser) -> None:
        """Test de parsing con texto antes y después del JSON."""
        response = '''Voy a resolver el problema paso a paso.

{
  "problem_type": "physics",
  "difficulty": "intermedio",
  "concepts": ["cinemática"],
  "solution": {
    "steps": [
      {"step_number": 1, "description": "Aplicar fórmula", "reasoning": "v = d/t"}
    ]
  },
  "final_answer": "100 km/h",
  "hints": [{"level": 1, "content": "Usa v=d/t", "concepts_referenced": []}]
}

Esa es mi respuesta.'''
        
        solution = parser.parse(response, "problema de física")
        
        assert solution.problem_type == ProblemType.PHYSICS
        assert solution.final_answer == "100 km/h"
    
    def test_parse_problem_type_mapping(self, parser: SolutionParser) -> None:
        """Test de mapeo de tipos de problema."""
        # Test varios aliases
        test_cases = [
            ("mathematics", ProblemType.MATHEMATICS),
            ("math", ProblemType.MATHEMATICS),
            ("física", ProblemType.PHYSICS),
            ("programming", ProblemType.PROGRAMMING),
            ("unknown", ProblemType.GENERAL),
        ]
        
        for type_str, expected in test_cases:
            result = parser._parse_problem_type(type_str)
            assert result == expected, f"Failed for {type_str}"
    
    def test_parse_difficulty_mapping(self, parser: SolutionParser) -> None:
        """Test de mapeo de dificultad."""
        test_cases = [
            ("básico", DifficultyLevel.BASIC),
            ("basic", DifficultyLevel.BASIC),
            ("intermedio", DifficultyLevel.INTERMEDIATE),
            ("avanzado", DifficultyLevel.ADVANCED),
            ("hard", DifficultyLevel.ADVANCED),
        ]
        
        for diff_str, expected in test_cases:
            result = parser._parse_difficulty(diff_str)
            assert result == expected, f"Failed for {diff_str}"
    
    def test_parse_hints_all_levels(self, parser: SolutionParser) -> None:
        """Test de parsing de pistas con todos los niveles."""
        hints_data = [
            {"level": 1, "content": "Sutil", "concepts_referenced": []},
            {"level": 2, "content": "Moderada", "concepts_referenced": []},
            {"level": 3, "content": "Directa", "concepts_referenced": []},
        ]
        
        hints = parser._parse_hints(hints_data)
        
        assert len(hints) == 3
        assert hints[0].level == HintLevel.SUBTLE
        assert hints[1].level == HintLevel.MODERATE
        assert hints[2].level == HintLevel.DIRECT
    
    def test_parse_hints_auto_complete(self, parser: SolutionParser) -> None:
        """Test de auto-completado de pistas faltantes."""
        # Solo nivel 1
        hints_data = [
            {"level": 1, "content": "Solo una pista", "concepts_referenced": []},
        ]
        
        hints = parser._parse_hints(hints_data)
        
        # Debe haber al menos 3 (una por nivel)
        assert len(hints) >= 3
        
        # Verificar que hay pistas de todos los niveles
        levels = {h.level for h in hints}
        assert HintLevel.SUBTLE in levels
        assert HintLevel.MODERATE in levels
        assert HintLevel.DIRECT in levels
    
    def test_parse_missing_required_field(self, parser: SolutionParser) -> None:
        """Test de error cuando falta campo requerido."""
        invalid_json = '''{
  "problem_type": "mathematics",
  "difficulty": "básico"
}'''
        
        with pytest.raises(SolutionParsingError) as exc_info:
            parser.parse(invalid_json, "test")
        
        assert "solution" in str(exc_info.value) or "final_answer" in str(exc_info.value)
    
    def test_parse_empty_steps(self, parser: SolutionParser) -> None:
        """Test de error cuando steps está vacío."""
        invalid_json = '''{
  "problem_type": "mathematics",
  "difficulty": "básico",
  "solution": {"steps": []},
  "final_answer": "42"
}'''
        
        with pytest.raises(SolutionParsingError, match="vacío"):
            parser.parse(invalid_json, "test")
    
    def test_parse_invalid_json(self, parser: SolutionParser) -> None:
        """Test de error con JSON malformado."""
        invalid_json = '{"problem_type": "math", "solution": {'
        
        with pytest.raises(JSONParsingError):
            parser.parse(invalid_json, "test")
    
    def test_extract_key_values(self, parser: SolutionParser) -> None:
        """Test de extracción de valores clave."""
        key_values = parser._extract_key_values("x = 42, y = 3.14")
        
        assert "42" in key_values
        assert "3.14" in key_values
    
    def test_extract_key_values_filters_trivial(self, parser: SolutionParser) -> None:
        """Test de que filtra valores triviales."""
        key_values = parser._extract_key_values("x = 0, y = 1, z = 42")
        
        assert "0" not in key_values
        assert "1" not in key_values
        assert "42" in key_values


class TestSolutionRepairStrategy:
    """Tests para SolutionRepairStrategy."""
    
    def test_repair_trailing_comma(self) -> None:
        """Test de reparación de trailing commas."""
        malformed = '{"a": 1, "b": 2,}'
        repaired = SolutionRepairStrategy.repair_json(malformed)
        
        assert ",}" not in repaired
    
    def test_repair_missing_quotes_on_keys(self) -> None:
        """Test de reparación de claves sin comillas."""
        malformed = '{problem_type: "math"}'
        repaired = SolutionRepairStrategy.repair_json(malformed)
        
        assert '"problem_type"' in repaired
    
    def test_create_minimal_solution(self) -> None:
        """Test de creación de solución mínima."""
        solution = SolutionRepairStrategy.create_minimal_solution(
            "Problema de prueba"
        )
        
        assert solution.problem_text == "Problema de prueba"
        assert solution.problem_type == ProblemType.GENERAL
        assert len(solution.steps) >= 1
        assert len(solution.hints) >= 1
        assert "[Respuesta no disponible" in solution.final_answer


class TestEdgeCases:
    """Tests para casos edge."""
    
    @pytest.fixture
    def parser(self) -> SolutionParser:
        return SolutionParser()
    
    def test_unicode_in_response(self, parser: SolutionParser) -> None:
        """Test con caracteres unicode."""
        response = '''{
  "problem_type": "mathematics",
  "difficulty": "básico",
  "concepts": ["álgebra", "ecuación"],
  "solution": {
    "steps": [
      {"step_number": 1, "description": "Solución π ≈ 3.14", "reasoning": "√2"}
    ]
  },
  "final_answer": "x = √2 ≈ 1.414",
  "hints": []
}'''
        
        solution = parser.parse(response, "test")
        
        assert "√2" in solution.final_answer
    
    def test_very_long_response(self, parser: SolutionParser) -> None:
        """Test con respuesta muy larga."""
        long_reasoning = "A" * 10000
        response = f'''{{
  "problem_type": "mathematics",
  "difficulty": "avanzado",
  "solution": {{
    "steps": [
      {{"step_number": 1, "description": "Paso largo", "reasoning": "{long_reasoning}"}}
    ]
  }},
  "final_answer": "42"
}}'''
        
        solution = parser.parse(response, "test")
        
        assert len(solution.steps[0].reasoning) == 10000
    
    def test_nested_json_in_calculation(self, parser: SolutionParser) -> None:
        """Test con JSON anidado en campo calculation."""
        response = '''{
  "problem_type": "programming",
  "difficulty": "intermedio",
  "solution": {
    "steps": [
      {
        "step_number": 1,
        "description": "Código",
        "reasoning": "Implementación",
        "calculation": "def solve(): return {'result': 42}"
      }
    ]
  },
  "final_answer": "{'result': 42}"
}'''
        
        solution = parser.parse(response, "test")
        
        assert "{'result': 42}" in solution.steps[0].calculation
