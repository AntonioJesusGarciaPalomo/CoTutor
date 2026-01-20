"""
Parser de soluciones estructuradas del Solver.

Este módulo maneja el parsing y validación de las respuestas JSON
generadas por el modelo del Solver.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.core.exceptions import JSONParsingError, SolutionParsingError
from src.core.types import (
    DifficultyLevel,
    Hint,
    HintLevel,
    ProblemType,
    SolutionStep,
    StructuredSolution,
)


class SolutionParser:
    """
    Parser para convertir respuestas JSON del LLM en StructuredSolution.
    
    Maneja varios casos comunes de formato incorrecto y proporciona
    validación robusta.
    """
    
    # Campos requeridos en la solución
    REQUIRED_FIELDS = ["problem_type", "solution", "final_answer"]
    
    # Campos requeridos en cada paso
    REQUIRED_STEP_FIELDS = ["step_number", "description"]
    
    def parse(self, raw_response: str, problem_text: str) -> StructuredSolution:
        """
        Parsea una respuesta raw del modelo a StructuredSolution.
        
        Args:
            raw_response: Respuesta raw del modelo (debería ser JSON).
            problem_text: Texto original del problema.
            
        Returns:
            StructuredSolution parseada y validada.
            
        Raises:
            JSONParsingError: Si el JSON no es válido.
            SolutionParsingError: Si faltan campos requeridos.
        """
        # Extraer JSON del response
        json_str = self._extract_json(raw_response)
        
        # Parsear JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise JSONParsingError(
                raw_response[:200],
                str(e),
            ) from e
        
        # Validar campos requeridos
        self._validate_required_fields(data)
        
        # Construir StructuredSolution
        return self._build_solution(data, problem_text)
    
    def _extract_json(self, response: str) -> str:
        """
        Extrae JSON de una respuesta que puede contener texto adicional.
        
        Maneja casos como:
        - JSON puro
        - JSON en bloques de código markdown
        - JSON con texto antes/después
        """
        response = response.strip()
        
        # Caso 1: Ya es JSON válido
        if response.startswith("{"):
            # Buscar el cierre del JSON
            return self._find_json_object(response)
        
        # Caso 2: JSON en bloque de código markdown
        code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        if matches:
            for match in matches:
                if match.strip().startswith("{"):
                    return match.strip()
        
        # Caso 3: Buscar JSON en cualquier parte del texto
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        # Buscar el JSON más grande (probablemente el correcto)
        if matches:
            return max(matches, key=len)
        
        # Caso 4: Intentar encontrar JSON incompleto y completarlo
        if "{" in response:
            start_idx = response.index("{")
            partial_json = response[start_idx:]
            
            # Contar llaves para detectar JSON incompleto
            open_braces = partial_json.count("{")
            close_braces = partial_json.count("}")
            
            if open_braces > close_braces:
                # Añadir llaves de cierre faltantes
                partial_json += "}" * (open_braces - close_braces)
                return partial_json
        
        raise JSONParsingError(response[:200], "No se encontró JSON válido en la respuesta")
    
    def _find_json_object(self, text: str) -> str:
        """Encuentra un objeto JSON completo al inicio del texto."""
        depth = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return text[:i+1]
        
        # Si llegamos aquí, el JSON está incompleto
        return text + "}" * depth
    
    def _validate_required_fields(self, data: dict[str, Any]) -> None:
        """Valida que estén presentes los campos requeridos."""
        missing = [field for field in self.REQUIRED_FIELDS if field not in data]
        
        if missing:
            raise SolutionParsingError(missing_fields=missing)
        
        # Validar estructura de solution
        solution = data.get("solution", {})
        if not isinstance(solution, dict):
            raise SolutionParsingError(cause="'solution' debe ser un objeto")
        
        steps = solution.get("steps", [])
        if not isinstance(steps, list):
            raise SolutionParsingError(cause="'solution.steps' debe ser una lista")
        
        if not steps:
            raise SolutionParsingError(cause="'solution.steps' está vacío")
        
        # Validar cada paso
        for i, step in enumerate(steps):
            missing_step_fields = [
                f for f in self.REQUIRED_STEP_FIELDS 
                if f not in step
            ]
            if missing_step_fields:
                raise SolutionParsingError(
                    cause=f"Paso {i+1} le faltan campos: {missing_step_fields}"
                )
    
    def _build_solution(
        self,
        data: dict[str, Any],
        problem_text: str,
    ) -> StructuredSolution:
        """Construye StructuredSolution desde el diccionario parseado."""
        
        # Parsear tipo de problema
        problem_type = self._parse_problem_type(data.get("problem_type", "general"))
        
        # Parsear dificultad
        difficulty = self._parse_difficulty(data.get("difficulty", "intermedio"))
        
        # Parsear pasos
        steps = self._parse_steps(data.get("solution", {}).get("steps", []))
        
        # Parsear pistas
        hints = self._parse_hints(data.get("hints", []))
        
        # Extraer key_values (valores que no deben revelarse)
        key_values = data.get("key_values", [])
        if not key_values:
            # Intentar extraer automáticamente del final_answer
            key_values = self._extract_key_values(data.get("final_answer", ""))
        
        return StructuredSolution(
            problem_text=problem_text,
            problem_type=problem_type,
            difficulty=difficulty,
            concepts=data.get("concepts", []),
            prerequisites=data.get("prerequisites", []),
            steps=steps,
            final_answer=str(data.get("final_answer", "")),
            verification=data.get("verification"),
            common_mistakes=data.get("common_mistakes", []),
            hints=hints,
            theory_references=data.get("theory_references", []),
            key_values=key_values,
        )
    
    def _parse_problem_type(self, type_str: str) -> ProblemType:
        """Convierte string a ProblemType enum."""
        type_mapping = {
            "mathematics": ProblemType.MATHEMATICS,
            "math": ProblemType.MATHEMATICS,
            "matemáticas": ProblemType.MATHEMATICS,
            "physics": ProblemType.PHYSICS,
            "física": ProblemType.PHYSICS,
            "chemistry": ProblemType.CHEMISTRY,
            "química": ProblemType.CHEMISTRY,
            "programming": ProblemType.PROGRAMMING,
            "programación": ProblemType.PROGRAMMING,
            "code": ProblemType.PROGRAMMING,
            "logic": ProblemType.LOGIC,
            "lógica": ProblemType.LOGIC,
        }
        
        return type_mapping.get(type_str.lower(), ProblemType.GENERAL)
    
    def _parse_difficulty(self, diff_str: str) -> DifficultyLevel:
        """Convierte string a DifficultyLevel enum."""
        diff_mapping = {
            "básico": DifficultyLevel.BASIC,
            "basico": DifficultyLevel.BASIC,
            "basic": DifficultyLevel.BASIC,
            "easy": DifficultyLevel.BASIC,
            "fácil": DifficultyLevel.BASIC,
            "intermedio": DifficultyLevel.INTERMEDIATE,
            "intermediate": DifficultyLevel.INTERMEDIATE,
            "medium": DifficultyLevel.INTERMEDIATE,
            "medio": DifficultyLevel.INTERMEDIATE,
            "avanzado": DifficultyLevel.ADVANCED,
            "advanced": DifficultyLevel.ADVANCED,
            "hard": DifficultyLevel.ADVANCED,
            "difícil": DifficultyLevel.ADVANCED,
        }
        
        return diff_mapping.get(diff_str.lower(), DifficultyLevel.INTERMEDIATE)
    
    def _parse_steps(self, steps_data: list[dict[str, Any]]) -> list[SolutionStep]:
        """Convierte lista de diccionarios a lista de SolutionStep."""
        steps = []
        
        for i, step_data in enumerate(steps_data):
            step = SolutionStep(
                step_number=step_data.get("step_number", i + 1),
                description=step_data.get("description", ""),
                reasoning=step_data.get("reasoning", ""),
                calculation=step_data.get("calculation"),
                result=step_data.get("result"),
                is_critical=step_data.get("is_critical", False),
            )
            steps.append(step)
        
        return steps
    
    def _parse_hints(self, hints_data: list[dict[str, Any]]) -> list[Hint]:
        """Convierte lista de diccionarios a lista de Hint."""
        hints = []
        
        for hint_data in hints_data:
            level_value = hint_data.get("level", 1)
            
            # Mapear nivel a enum
            if level_value == 1:
                level = HintLevel.SUBTLE
            elif level_value == 2:
                level = HintLevel.MODERATE
            else:
                level = HintLevel.DIRECT
            
            hint = Hint(
                level=level,
                content=hint_data.get("content", ""),
                concepts_referenced=hint_data.get("concepts_referenced", []),
            )
            hints.append(hint)
        
        # Asegurar que hay al menos una pista por nivel
        existing_levels = {h.level for h in hints}
        
        if HintLevel.SUBTLE not in existing_levels:
            hints.insert(0, Hint(
                level=HintLevel.SUBTLE,
                content="¿Qué conceptos crees que podrían aplicarse a este problema?",
                concepts_referenced=[],
            ))
        
        if HintLevel.MODERATE not in existing_levels:
            hints.append(Hint(
                level=HintLevel.MODERATE,
                content="Piensa en los pasos fundamentales para abordar este tipo de problema.",
                concepts_referenced=[],
            ))
        
        if HintLevel.DIRECT not in existing_levels:
            hints.append(Hint(
                level=HintLevel.DIRECT,
                content="Revisa el método paso a paso para este tipo de ejercicio.",
                concepts_referenced=[],
            ))
        
        return sorted(hints, key=lambda h: h.level.value)
    
    def _extract_key_values(self, final_answer: str) -> list[str]:
        """Extrae valores numéricos clave de la respuesta final."""
        # Buscar números en la respuesta
        numbers = re.findall(r'-?\d+\.?\d*', final_answer)
        
        # Filtrar valores triviales
        key_values = [
            n for n in numbers 
            if n not in ('0', '1', '-1', '0.0', '1.0')
        ]
        
        return key_values[:10]  # Limitar a 10 valores


class SolutionRepairStrategy:
    """
    Estrategias para reparar soluciones incompletas o malformadas.
    """
    
    @staticmethod
    def repair_json(malformed_json: str) -> str:
        """
        Intenta reparar JSON malformado.
        
        Maneja casos comunes como:
        - Comillas no escapadas
        - Trailing commas
        - Llaves faltantes
        """
        text = malformed_json
        
        # Eliminar trailing commas antes de } o ]
        text = re.sub(r',\s*([}\]])', r'\1', text)
        
        # Escapar comillas internas en strings (heurística simple)
        # Esto es complejo de hacer bien, así que solo manejamos casos simples
        
        # Asegurar que las claves tienen comillas
        text = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
        
        return text
    
    @staticmethod
    def create_minimal_solution(
        problem_text: str,
        partial_data: dict[str, Any] | None = None,
    ) -> StructuredSolution:
        """
        Crea una solución mínima cuando el parsing falla completamente.
        
        Esta solución indica que se necesita procesar de nuevo.
        """
        return StructuredSolution(
            problem_text=problem_text,
            problem_type=ProblemType.GENERAL,
            difficulty=DifficultyLevel.INTERMEDIATE,
            concepts=[],
            prerequisites=[],
            steps=[
                SolutionStep(
                    step_number=1,
                    description="Solución pendiente de generación",
                    reasoning="El parser no pudo procesar la respuesta del modelo",
                    is_critical=True,
                )
            ],
            final_answer="[Respuesta no disponible - requiere regeneración]",
            verification=None,
            common_mistakes=[],
            hints=[
                Hint(
                    level=HintLevel.SUBTLE,
                    content="Analiza el problema con cuidado",
                    concepts_referenced=[],
                ),
            ],
            theory_references=[],
            key_values=[],
        )


# Instancia global del parser
solution_parser = SolutionParser()


__all__ = [
    "SolutionParser",
    "SolutionRepairStrategy",
    "solution_parser",
]
