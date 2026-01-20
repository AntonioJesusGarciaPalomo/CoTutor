"""
Detectores del sistema de guardrails.

Este módulo expone los detectores principales:
- ManipulationDetector: Detecta intentos de manipulación
- SolutionLeakDetector: Detecta fugas de solución
- PedagogicalValidator: Valida calidad pedagógica
"""

from src.guardrails.detectors.manipulation import ManipulationDetector
from src.guardrails.detectors.pedagogical import PedagogicalValidator
from src.guardrails.detectors.solution_leak import SolutionLeakDetector


__all__ = [
    "ManipulationDetector",
    "SolutionLeakDetector",
    "PedagogicalValidator",
]
