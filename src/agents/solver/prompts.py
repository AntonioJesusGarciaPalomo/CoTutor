"""
Prompts para el Agente Solucionador.

Este módulo contiene los prompts del sistema para el Solver,
diseñados para generar soluciones estructuradas en formato JSON.
"""

from __future__ import annotations

from string import Template
from typing import Any

# =============================================================================
# PROMPT PRINCIPAL DEL SOLVER
# =============================================================================

SOLVER_SYSTEM_PROMPT = """Eres un experto solucionador de problemas educativos. Resuelve el problema paso a paso.

REGLAS:
1. Resuelve el problema COMPLETAMENTE
2. Muestra cada paso con su razonamiento profundo
3. Da la respuesta final claramente y genera pistas útiles

FORMATO DE SALIDA (ESTRICTAMENTE JSON, sin texto fuera del JSON):
{
  "problem_type": "mathematics",
  "difficulty": "intermedio",
  "concepts": ["concepto_1", "concepto_2"],
  "prerequisites": ["prerequisito_1"],
  "solution": {
    "steps": [
      {
        "step_number": 1,
        "description": "Explicación del paso",
        "reasoning": "Por qué hacemos esto",
        "calculation": "Ecuación o código",
        "result": "Resultado intermedio",
        "is_critical": true
      }
    ]
  },
  "final_answer": "La respuesta final",
  "verification": "Cómo verificar el resultado",
  "common_mistakes": ["error común 1"],
  "hints": [
    {"level": 1, "content": "Pista muy sutil", "concepts_referenced": []},
    {"level": 2, "content": "Pista moderada", "concepts_referenced": []},
    {"level": 3, "content": "Pista muy directa", "concepts_referenced": []}
  ],
  "theory_references": ["referencia 1"],
  "key_values": ["valor1", "valor2"]
}

Cada step DEBE tener: step_number, description, reasoning, is_critical.
Campos opcionales en steps: calculation, result.
Solo responde con el JSON válido, sin bloques de código markdown extra y sin texto explicativo fuera del JSON."""


# =============================================================================
# PROMPTS ESPECIALIZADOS POR DOMINIO
# =============================================================================

MATH_SOLVER_SYSTEM_PROMPT = """Eres un experto matemático. Muestra TODOS los pasos algebraicos, justifica cada transformación y verifica sustituyendo en la ecuación original.

""" + SOLVER_SYSTEM_PROMPT


PHYSICS_SOLVER_SYSTEM_PROMPT = """Eres un experto físico. Lista variables con UNIDADES, selecciona ecuaciones apropiadas, resuelve algebraicamente antes de sustituir números, y verifica unidades.

""" + SOLVER_SYSTEM_PROMPT


PROGRAMMING_SOLVER_SYSTEM_PROMPT = """Eres un experto programador. Identifica casos edge, diseña el algoritmo, implementa con código limpio, y analiza complejidad temporal/espacial. Usa el campo "calculation" para código.

""" + SOLVER_SYSTEM_PROMPT


CHEMISTRY_SOLVER_SYSTEM_PROMPT = """Eres un experto químico. Balancea ecuaciones, identifica reactivo limitante, usa factores de conversión con unidades, y verifica conservación de masa y carga.

""" + SOLVER_SYSTEM_PROMPT


# =============================================================================
# TEMPLATES PARA CONSTRUIR PROMPTS
# =============================================================================

PROBLEM_TEMPLATE = Template("""
PROBLEMA A RESOLVER:
$problem_text

$additional_context

Resuelve este problema completamente y genera la estructura JSON especificada.
""")


FOLLOW_UP_TEMPLATE = Template("""
PROBLEMA ORIGINAL:
$problem_text

SOLUCIÓN ANTERIOR (parcial o con errores):
$previous_solution

FEEDBACK:
$feedback

Por favor, revisa y corrige la solución, generando el JSON completo actualizado.
""")


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def get_solver_prompt(domain: str = "general") -> str:
    """
    Obtiene el prompt del solver para un dominio específico.
    
    Args:
        domain: Dominio del problema (mathematics, physics, programming, chemistry, general)
        
    Returns:
        Prompt del sistema apropiado para el dominio.
    """
    prompts = {
        "mathematics": MATH_SOLVER_SYSTEM_PROMPT,
        "math": MATH_SOLVER_SYSTEM_PROMPT,
        "physics": PHYSICS_SOLVER_SYSTEM_PROMPT,
        "programming": PROGRAMMING_SOLVER_SYSTEM_PROMPT,
        "code": PROGRAMMING_SOLVER_SYSTEM_PROMPT,
        "chemistry": CHEMISTRY_SOLVER_SYSTEM_PROMPT,
        "general": SOLVER_SYSTEM_PROMPT,
    }
    
    return prompts.get(domain.lower(), SOLVER_SYSTEM_PROMPT)


def format_problem_prompt(
    problem_text: str,
    additional_context: str = "",
) -> str:
    """
    Formatea el prompt del problema.
    
    Args:
        problem_text: Texto del problema a resolver.
        additional_context: Contexto adicional opcional.
        
    Returns:
        Prompt formateado.
    """
    return PROBLEM_TEMPLATE.substitute(
        problem_text=problem_text,
        additional_context=additional_context,
    )


def format_followup_prompt(
    problem_text: str,
    previous_solution: str,
    feedback: str,
) -> str:
    """
    Formatea un prompt de seguimiento para corregir errores.
    
    Args:
        problem_text: Texto del problema original.
        previous_solution: Solución anterior (con errores).
        feedback: Feedback sobre qué corregir.
        
    Returns:
        Prompt de seguimiento formateado.
    """
    return FOLLOW_UP_TEMPLATE.substitute(
        problem_text=problem_text,
        previous_solution=previous_solution,
        feedback=feedback,
    )


# =============================================================================
# EJEMPLOS DE SOLUCIONES (para few-shot learning si es necesario)
# =============================================================================

EXAMPLE_MATH_SOLUTION = """{
  "problem_type": "mathematics",
  "difficulty": "básico",
  "concepts": ["ecuaciones lineales", "despeje de variables"],
  "prerequisites": ["operaciones básicas", "propiedades de igualdad"],
  "solution": {
    "steps": [
      {
        "step_number": 1,
        "description": "Identificar la ecuación y la variable a despejar",
        "reasoning": "Tenemos una ecuación lineal con una incógnita x",
        "calculation": "2x + 3 = 7",
        "result": "Ecuación identificada",
        "is_critical": false
      },
      {
        "step_number": 2,
        "description": "Restar 3 de ambos lados",
        "reasoning": "Para aislar el término con x, restamos el término independiente",
        "calculation": "2x + 3 - 3 = 7 - 3",
        "result": "2x = 4",
        "is_critical": true
      },
      {
        "step_number": 3,
        "description": "Dividir ambos lados entre 2",
        "reasoning": "Para obtener x sola, dividimos entre el coeficiente",
        "calculation": "2x/2 = 4/2",
        "result": "x = 2",
        "is_critical": true
      }
    ]
  },
  "final_answer": "x = 2",
  "verification": "Sustituyendo x=2 en la ecuación original: 2(2) + 3 = 4 + 3 = 7 ✓",
  "common_mistakes": [
    "Olvidar aplicar la operación a ambos lados de la igualdad",
    "Confundir el orden de operaciones al despejar",
    "Errores de signo al restar"
  ],
  "hints": [
    {
      "level": 1,
      "content": "¿Qué operaciones necesitas para aislar la variable?",
      "concepts_referenced": ["despeje de variables"]
    },
    {
      "level": 2,
      "content": "Primero elimina el término que está sumando a la x",
      "concepts_referenced": ["operaciones inversas"]
    },
    {
      "level": 3,
      "content": "Resta 3 de ambos lados, luego divide entre el coeficiente de x",
      "concepts_referenced": ["ecuaciones lineales"]
    }
  ],
  "theory_references": [
    "Propiedad de la igualdad: si a = b, entonces a + c = b + c",
    "Operaciones inversas: suma/resta, multiplicación/división"
  ],
  "key_values": ["2", "4", "7", "3"]
}"""


__all__ = [
    "SOLVER_SYSTEM_PROMPT",
    "MATH_SOLVER_SYSTEM_PROMPT",
    "PHYSICS_SOLVER_SYSTEM_PROMPT",
    "PROGRAMMING_SOLVER_SYSTEM_PROMPT",
    "CHEMISTRY_SOLVER_SYSTEM_PROMPT",
    "get_solver_prompt",
    "format_problem_prompt",
    "format_followup_prompt",
    "EXAMPLE_MATH_SOLUTION",
]
