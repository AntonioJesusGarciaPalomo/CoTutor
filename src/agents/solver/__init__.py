"""
Módulo del Agente Solucionador.

El Solver Agent es responsable de:
- Recibir problemas educativos
- Resolverlos completamente
- Generar soluciones estructuradas con pasos
- Crear pistas progresivas para el Tutor
- Identificar errores comunes

Ejemplo de uso:
    ```python
    from src.agents.solver import SolverAgent
    
    # Crear agente
    solver = await SolverAgent.create("ollama/qwen2.5:14b")
    
    # Resolver un problema
    solution = await solver.solve("Resuelve: 2x + 3 = 7")
    
    print(solution.final_answer)  # "x = 2"
    print(solution.steps)         # Pasos de la solución
    print(solution.hints)         # Pistas para el tutor
    ```
"""

from src.agents.solver.agent import SolutionCache, SolverAgent
from src.agents.solver.parser import SolutionParser, SolutionRepairStrategy, solution_parser
from src.agents.solver.prompts import (
    CHEMISTRY_SOLVER_SYSTEM_PROMPT,
    MATH_SOLVER_SYSTEM_PROMPT,
    PHYSICS_SOLVER_SYSTEM_PROMPT,
    PROGRAMMING_SOLVER_SYSTEM_PROMPT,
    SOLVER_SYSTEM_PROMPT,
    format_followup_prompt,
    format_problem_prompt,
    get_solver_prompt,
)
from src.agents.solver.tools import (
    ProblemClassifier,
    SafeCalculator,
    SolutionValidator,
    calculator,
    classifier,
    validator,
)

__all__ = [
    # Agent
    "SolverAgent",
    "SolutionCache",
    # Parser
    "SolutionParser",
    "SolutionRepairStrategy",
    "solution_parser",
    # Prompts
    "SOLVER_SYSTEM_PROMPT",
    "MATH_SOLVER_SYSTEM_PROMPT",
    "PHYSICS_SOLVER_SYSTEM_PROMPT",
    "PROGRAMMING_SOLVER_SYSTEM_PROMPT",
    "CHEMISTRY_SOLVER_SYSTEM_PROMPT",
    "get_solver_prompt",
    "format_problem_prompt",
    "format_followup_prompt",
    # Tools
    "SafeCalculator",
    "SolutionValidator",
    "ProblemClassifier",
    "calculator",
    "validator",
    "classifier",
]
