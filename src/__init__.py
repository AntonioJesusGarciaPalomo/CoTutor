"""
Aula AI Tutor - Sistema de Agentes Educativos.

Un sistema multi-agente para tutoría educativa usando el método socrático.
El sistema consta de dos agentes principales:

1. **Solver Agent**: Resuelve problemas y genera soluciones estructuradas.
2. **Tutor Agent**: Guía al estudiante usando el método socrático, sin revelar
   la solución directamente.

Los agentes se comunican via el protocolo A2A (Agent-to-Agent).

Módulos principales:
- `core`: Tipos, estructuras de datos y excepciones
- `models`: Capa de abstracción para modelos de lenguaje
- `agents`: Implementación de agentes (solver, tutor)
- `guardrails`: Sistema de filtros para seguridad y pedagogía
- `a2a`: Protocolo de comunicación entre agentes
- `utils`: Logging, métricas y utilidades

Ejemplo de uso rápido:
    ```python
    from src.models import get_model
    
    # Obtener modelo para el Solver
    solver_model = await get_model("ollama/qwen2.5:14b")
    
    # Generar respuesta
    response = await solver_model.generate([
        {"role": "system", "content": "Eres un experto en matemáticas..."},
        {"role": "user", "content": "Resuelve: 2x + 3 = 7"}
    ])
    
    print(response.content)
    ```
"""

__version__ = "0.1.0"
__author__ = "ThaleOn AI Systems"
