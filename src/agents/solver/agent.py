"""
Agente Solucionador (Solver Agent).

Este agente recibe problemas educativos y genera soluciones estructuradas
completas que serán usadas por el agente Tutor para guiar a los estudiantes.

El Solver:
- Resuelve el problema completamente
- Estructura la solución en pasos
- Genera pistas progresivas
- Identifica errores comunes
- Proporciona referencias teóricas
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from src.core.exceptions import (
    ModelGenerationError,
    SolutionParsingError,
    SolverError,
)
from src.core.types import (
    DifficultyLevel,
    HintLevel,
    Message,
    MessageRole,
    ModelResponse,
    ProblemType,
    StructuredSolution,
)
from src.models.base import BaseModelAdapter
from src.models.factory import get_model
from src.utils.logging import get_logger
from src.utils.metrics import get_metrics

from .parser import SolutionParser, SolutionRepairStrategy, solution_parser
from .prompts import (
    format_followup_prompt,
    format_problem_prompt,
    get_solver_prompt,
)
from .tools import ProblemClassifier, classifier


logger = get_logger(__name__)
metrics = get_metrics()


class SolutionCache:
    """
    Caché en memoria para soluciones generadas.
    
    Evita regenerar soluciones para el mismo problema.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600) -> None:
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache: dict[str, tuple[StructuredSolution, datetime]] = {}
    
    def _make_key(self, problem_text: str) -> str:
        """Genera una clave única para el problema."""
        # Normalizar texto y generar hash
        normalized = " ".join(problem_text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]
    
    def get(self, problem_text: str) -> StructuredSolution | None:
        """Obtiene una solución cacheada si existe y no ha expirado."""
        key = self._make_key(problem_text)
        
        if key in self._cache:
            solution, timestamp = self._cache[key]
            if datetime.now() - timestamp < self.ttl:
                metrics.increment("solver_cache_hits")
                return solution
            else:
                # Expirado, eliminar
                del self._cache[key]
        
        metrics.increment("solver_cache_misses")
        return None
    
    def set(self, problem_text: str, solution: StructuredSolution) -> None:
        """Almacena una solución en caché."""
        # Limpiar entradas expiradas si estamos llenos
        if len(self._cache) >= self.max_size:
            self._cleanup()
        
        key = self._make_key(problem_text)
        self._cache[key] = (solution, datetime.now())
    
    def _cleanup(self) -> None:
        """Elimina entradas expiradas."""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if now - timestamp >= self.ttl
        ]
        for key in expired_keys:
            del self._cache[key]
        
        # Si aún estamos llenos, eliminar las más antiguas
        if len(self._cache) >= self.max_size:
            sorted_items = sorted(
                self._cache.items(),
                key=lambda x: x[1][1]
            )
            # Eliminar el 25% más antiguo
            to_remove = len(sorted_items) // 4
            for key, _ in sorted_items[:to_remove]:
                del self._cache[key]
    
    def clear(self) -> None:
        """Limpia toda la caché."""
        self._cache.clear()
    
    def __len__(self) -> int:
        return len(self._cache)


class SolverAgent:
    """
    Agente Solucionador para problemas educativos.
    
    Este agente es el "cerebro" que resuelve problemas y genera
    soluciones estructuradas que el Tutor usa para guiar estudiantes.
    
    Attributes:
        model: Adaptador del modelo de lenguaje.
        parser: Parser de soluciones JSON.
        classifier: Clasificador de tipos de problemas.
        cache: Caché de soluciones.
    
    Example:
        ```python
        # Crear agente
        solver = await SolverAgent.create("ollama/qwen2.5:14b")
        
        # Resolver un problema
        solution = await solver.solve(
            "Resuelve la ecuación: 2x + 3 = 7"
        )
        
        print(solution.final_answer)  # "x = 2"
        print(solution.steps)         # Lista de pasos
        print(solution.hints)         # Pistas para el tutor
        ```
    """
    
    def __init__(
        self,
        model: BaseModelAdapter,
        parser: SolutionParser | None = None,
        problem_classifier: ProblemClassifier | None = None,
        use_cache: bool = True,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        max_retries: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        """
        Inicializa el Solver Agent.
        
        Args:
            model: Adaptador del modelo de lenguaje.
            parser: Parser de soluciones (opcional).
            problem_classifier: Clasificador de problemas (opcional).
            use_cache: Si usar caché de soluciones.
            cache_size: Tamaño máximo de la caché.
            cache_ttl: Tiempo de vida en caché (segundos).
            max_retries: Máximo de reintentos en caso de error.
            temperature: Temperatura del modelo (baja para precisión).
            max_tokens: Máximo de tokens a generar.
        """
        self.model = model
        self.parser = parser or solution_parser
        self.classifier = problem_classifier or classifier
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Caché opcional
        self.use_cache = use_cache
        self.cache = SolutionCache(cache_size, cache_ttl) if use_cache else None
        
        logger.info(
            "SolverAgent inicializado",
            model=model.model_id,
            use_cache=use_cache,
        )
    
    @classmethod
    async def create(
        cls,
        model_id: str = "ollama/qwen2.5:14b",
        **kwargs: Any,
    ) -> "SolverAgent":
        """
        Factory method para crear un SolverAgent.
        
        Args:
            model_id: Identificador del modelo a usar.
            **kwargs: Argumentos adicionales para el constructor.
            
        Returns:
            SolverAgent configurado y listo.
        """
        model = await get_model(model_id)
        return cls(model=model, **kwargs)
    
    async def solve(
        self,
        problem_text: str,
        domain_hint: str | None = None,
        additional_context: str = "",
        force_regenerate: bool = False,
    ) -> StructuredSolution:
        """
        Resuelve un problema y genera una solución estructurada.
        
        Args:
            problem_text: Texto del problema a resolver.
            domain_hint: Dominio sugerido (mathematics, physics, etc.).
            additional_context: Contexto adicional para el problema.
            force_regenerate: Forzar regeneración ignorando caché.
            
        Returns:
            StructuredSolution con la solución completa.
            
        Raises:
            SolverError: Si no se puede generar una solución válida.
        """
        start_time = time.perf_counter()
        
        # Verificar caché
        if self.use_cache and self.cache and not force_regenerate:
            cached = self.cache.get(problem_text)
            if cached:
                logger.debug("Solución obtenida de caché", problem_preview=problem_text[:50])
                return cached
        
        # Clasificar el problema si no se dio hint
        if domain_hint:
            domain = domain_hint
        else:
            domain, confidence = self.classifier.classify(problem_text)
            logger.debug(
                "Problema clasificado",
                domain=domain,
                confidence=confidence,
            )
        
        # Obtener prompt apropiado
        system_prompt = get_solver_prompt(domain)
        user_prompt = format_problem_prompt(problem_text, additional_context)
        
        # Generar solución con reintentos
        solution = await self._generate_with_retries(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            problem_text=problem_text,
        )
        
        # Asignar modelo usado
        solution.solver_model = self.model.model_id
        
        # Guardar en caché
        if self.use_cache and self.cache:
            self.cache.set(problem_text, solution)
        
        # Métricas
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics.observe("solver_generation_time_ms", elapsed_ms)
        metrics.increment("solver_problems_solved", labels={"domain": domain})
        
        logger.info(
            "Problema resuelto",
            domain=domain,
            difficulty=solution.difficulty.value,
            steps_count=len(solution.steps),
            elapsed_ms=elapsed_ms,
        )
        
        return solution
    
    async def _generate_with_retries(
        self,
        system_prompt: str,
        user_prompt: str,
        problem_text: str,
    ) -> StructuredSolution:
        """Genera solución con reintentos en caso de error."""
        last_error: Exception | None = None
        last_response: str = ""
        
        for attempt in range(self.max_retries):
            try:
                # Construir mensajes
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                
                # Si tenemos un error previo, añadir feedback
                if last_error and last_response:
                    feedback_prompt = self._create_feedback_prompt(
                        last_response, str(last_error)
                    )
                    messages.append({"role": "assistant", "content": last_response})
                    messages.append({"role": "user", "content": feedback_prompt})
                
                # Generar respuesta
                with metrics.timer("solver_model_call_ms"):
                    response = await self.model.generate(
                        messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                
                last_response = response.content
                
                # Parsear solución
                solution = self.parser.parse(response.content, problem_text)
                
                # Validar calidad de la solución
                self._validate_solution_quality(solution)
                
                return solution
                
            except (SolutionParsingError, ValueError) as e:
                last_error = e
                logger.warning(
                    "Error parseando solución, reintentando",
                    attempt=attempt + 1,
                    error=str(e),
                )
                metrics.increment("solver_parse_errors")
                
            except ModelGenerationError as e:
                last_error = e
                logger.warning(
                    "Error generando solución",
                    attempt=attempt + 1,
                    error=str(e),
                )
                metrics.increment("solver_generation_errors")
                
                # Esperar antes de reintentar
                await asyncio.sleep(1.0 * (attempt + 1))
        
        # Todos los reintentos fallaron
        logger.error(
            "Fallo al resolver problema después de reintentos",
            max_retries=self.max_retries,
            last_error=str(last_error),
        )
        
        # Intentar crear una solución mínima
        if last_response:
            try:
                return SolutionRepairStrategy.create_minimal_solution(
                    problem_text
                )
            except Exception:
                pass
        
        raise SolverError(
            f"No se pudo generar solución después de {self.max_retries} intentos: {last_error}",
            problem_preview=problem_text[:200],
        )
    
    def _create_feedback_prompt(self, last_response: str, error: str) -> str:
        """Crea un prompt de feedback para corregir errores."""
        return f"""Tu respuesta anterior tuvo un error:
{error}

Tu respuesta fue:
{last_response[:500]}...

Por favor, genera de nuevo el JSON completo con la solución, 
asegurándote de que sea JSON válido y tenga todos los campos requeridos."""
    
    def _validate_solution_quality(self, solution: StructuredSolution) -> None:
        """
        Valida la calidad de una solución generada.
        
        Raises:
            ValueError: Si la solución no cumple criterios mínimos.
        """
        # Debe tener al menos un paso
        if not solution.steps:
            raise ValueError("La solución debe tener al menos un paso")
        
        # La respuesta final no debe estar vacía
        if not solution.final_answer or solution.final_answer.strip() == "":
            raise ValueError("La respuesta final no puede estar vacía")
        
        # Debe tener al menos una pista
        if not solution.hints:
            raise ValueError("La solución debe incluir pistas")
        
        # Los pasos deben tener descripción
        for step in solution.steps:
            if not step.description:
                raise ValueError(f"El paso {step.step_number} no tiene descripción")
    
    async def solve_with_verification(
        self,
        problem_text: str,
        expected_answer: str | None = None,
        **kwargs: Any,
    ) -> tuple[StructuredSolution, bool]:
        """
        Resuelve y opcionalmente verifica contra una respuesta esperada.
        
        Args:
            problem_text: Texto del problema.
            expected_answer: Respuesta esperada para verificación.
            **kwargs: Argumentos adicionales para solve().
            
        Returns:
            Tupla (solución, coincide_con_esperada).
        """
        solution = await self.solve(problem_text, **kwargs)
        
        if expected_answer is None:
            return solution, True
        
        # Verificar si la respuesta coincide
        # Normalizar ambas respuestas para comparación
        generated = self._normalize_answer(solution.final_answer)
        expected = self._normalize_answer(expected_answer)
        
        matches = generated == expected
        
        if not matches:
            logger.warning(
                "Respuesta no coincide con esperada",
                generated=generated,
                expected=expected,
            )
            metrics.increment("solver_answer_mismatches")
        
        return solution, matches
    
    def _normalize_answer(self, answer: str) -> str:
        """Normaliza una respuesta para comparación."""
        # Eliminar espacios y convertir a minúsculas
        normalized = " ".join(answer.lower().split())
        
        # Eliminar signos de puntuación al final
        normalized = normalized.rstrip(".,;:")
        
        # Normalizar formatos de variables comunes
        # x = 2 -> x=2
        import re
        normalized = re.sub(r'\s*=\s*', '=', normalized)
        
        return normalized
    
    async def get_hints_for_level(
        self,
        solution: StructuredSolution,
        level: HintLevel,
    ) -> list[str]:
        """
        Obtiene pistas hasta un cierto nivel.
        
        Args:
            solution: Solución estructurada.
            level: Nivel máximo de pistas.
            
        Returns:
            Lista de contenidos de pistas.
        """
        hints = solution.get_hints_for_level(level)
        return [h.content for h in hints]
    
    async def regenerate_hints(
        self,
        problem_text: str,
        current_solution: StructuredSolution,
        student_attempt: str,
    ) -> list[str]:
        """
        Regenera pistas basadas en el intento del estudiante.
        
        Esto permite generar pistas más específicas según
        dónde parece estar atascado el estudiante.
        
        Args:
            problem_text: Problema original.
            current_solution: Solución actual.
            student_attempt: Intento del estudiante.
            
        Returns:
            Lista de nuevas pistas personalizadas.
        """
        prompt = f"""Dado el siguiente problema y el intento del estudiante,
genera 3 pistas específicas que le ayuden sin revelar la solución.

PROBLEMA:
{problem_text}

INTENTO DEL ESTUDIANTE:
{student_attempt}

RESPUESTA CORRECTA (NO REVELAR):
{current_solution.final_answer}

Genera exactamente 3 pistas en formato JSON:
[
  {{"level": 1, "content": "pista sutil"}},
  {{"level": 2, "content": "pista moderada"}},
  {{"level": 3, "content": "pista directa sin dar la respuesta"}}
]"""
        
        response = await self.model.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        
        try:
            import json
            hints_data = json.loads(response.content)
            return [h["content"] for h in hints_data]
        except Exception:
            # Fallback a pistas genéricas
            return [h.content for h in current_solution.hints[:3]]
    
    def get_cache_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas de la caché."""
        if not self.cache:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "size": len(self.cache),
            "max_size": self.cache.max_size,
            "ttl_seconds": self.cache.ttl.total_seconds(),
        }
    
    def clear_cache(self) -> None:
        """Limpia la caché de soluciones."""
        if self.cache:
            self.cache.clear()
            logger.info("Caché del solver limpiada")


__all__ = [
    "SolverAgent",
    "SolutionCache",
]
