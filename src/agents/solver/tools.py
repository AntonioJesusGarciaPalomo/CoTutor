"""
Herramientas auxiliares para el Agente Solucionador.

Este módulo proporciona herramientas que el Solver puede usar
para verificar cálculos, validar expresiones matemáticas, etc.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from decimal import Decimal, InvalidOperation
from typing import Any


# =============================================================================
# CALCULADORA SEGURA
# =============================================================================

class SafeCalculator:
    """
    Calculadora segura que evalúa expresiones matemáticas sin exec/eval.
    
    Soporta operaciones básicas y funciones matemáticas comunes,
    pero no permite código arbitrario.
    
    Example:
        ```python
        calc = SafeCalculator()
        result = calc.evaluate("2 * (3 + 4)")  # 14
        result = calc.evaluate("sqrt(16)")     # 4.0
        result = calc.evaluate("sin(pi/2)")    # 1.0
        ```
    """
    
    # Operadores soportados
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # Funciones matemáticas permitidas
    FUNCTIONS = {
        # Básicas
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        
        # Trigonométricas
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        
        # Exponenciales y logarítmicas
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "sqrt": math.sqrt,
        "pow": math.pow,
        
        # Otras
        "ceil": math.ceil,
        "floor": math.floor,
        "factorial": math.factorial,
        "gcd": math.gcd,
        "degrees": math.degrees,
        "radians": math.radians,
    }
    
    # Constantes permitidas
    CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
    }
    
    def evaluate(self, expression: str) -> float | int:
        """
        Evalúa una expresión matemática de forma segura.
        
        Args:
            expression: Expresión matemática como string.
            
        Returns:
            Resultado de la evaluación.
            
        Raises:
            ValueError: Si la expresión es inválida o contiene operaciones no permitidas.
        """
        # Limpiar expresión
        expression = expression.strip()
        
        # Reemplazar constantes
        for name, value in self.CONSTANTS.items():
            expression = re.sub(rf'\b{name}\b', str(value), expression)
        
        try:
            tree = ast.parse(expression, mode='eval')
            return self._eval_node(tree.body)
        except (SyntaxError, TypeError) as e:
            raise ValueError(f"Expresión inválida: {expression}") from e
    
    def _eval_node(self, node: ast.AST) -> float | int:
        """Evalúa recursivamente un nodo del AST."""
        
        # Números
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Constante no soportada: {node.value}")
        
        # Operaciones binarias
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_func = self.OPERATORS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Operador no soportado: {type(node.op).__name__}")
            return op_func(left, right)
        
        # Operaciones unarias
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_func = self.OPERATORS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Operador unario no soportado: {type(node.op).__name__}")
            return op_func(operand)
        
        # Llamadas a funciones
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Solo se permiten funciones simples")
            
            func_name = node.func.id
            if func_name not in self.FUNCTIONS:
                raise ValueError(f"Función no permitida: {func_name}")
            
            args = [self._eval_node(arg) for arg in node.args]
            return self.FUNCTIONS[func_name](*args)
        
        # Variables (solo constantes predefinidas)
        if isinstance(node, ast.Name):
            if node.id in self.CONSTANTS:
                return self.CONSTANTS[node.id]
            raise ValueError(f"Variable no definida: {node.id}")
        
        raise ValueError(f"Tipo de nodo no soportado: {type(node).__name__}")
    
    def validate_expression(self, expression: str) -> tuple[bool, str | None]:
        """
        Valida si una expresión es evaluable.
        
        Args:
            expression: Expresión a validar.
            
        Returns:
            Tupla (es_válida, mensaje_error).
        """
        try:
            self.evaluate(expression)
            return True, None
        except ValueError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Error inesperado: {e}"


# =============================================================================
# VALIDADOR DE SOLUCIONES
# =============================================================================

class SolutionValidator:
    """
    Validador para verificar soluciones matemáticas.
    
    Permite verificar si una solución satisface una ecuación
    o cumple ciertas condiciones.
    """
    
    def __init__(self) -> None:
        self.calculator = SafeCalculator()
    
    def verify_equation_solution(
        self,
        equation: str,
        variable: str,
        value: float,
        tolerance: float = 1e-9,
    ) -> tuple[bool, float]:
        """
        Verifica si un valor es solución de una ecuación.
        
        Args:
            equation: Ecuación en forma "expresion1 = expresion2".
            variable: Nombre de la variable (ej: "x").
            value: Valor propuesto para la variable.
            tolerance: Tolerancia para comparación de flotantes.
            
        Returns:
            Tupla (es_solución, diferencia).
        """
        # Separar la ecuación en dos lados
        if "=" not in equation:
            raise ValueError("La ecuación debe contener '='")
        
        left, right = equation.split("=", 1)
        
        # Sustituir la variable por el valor
        left = re.sub(rf'\b{variable}\b', str(value), left)
        right = re.sub(rf'\b{variable}\b', str(value), right)
        
        # Evaluar ambos lados
        left_value = self.calculator.evaluate(left)
        right_value = self.calculator.evaluate(right)
        
        # Calcular diferencia
        diff = abs(left_value - right_value)
        
        return diff <= tolerance, diff
    
    def verify_inequality(
        self,
        expression: str,
        variable: str,
        value: float,
    ) -> bool:
        """
        Verifica si un valor satisface una desigualdad.
        
        Args:
            expression: Desigualdad (ej: "x > 0", "2*x <= 10").
            variable: Nombre de la variable.
            value: Valor a verificar.
            
        Returns:
            True si el valor satisface la desigualdad.
        """
        # Detectar el operador de comparación
        operators_map = {
            ">=": operator.ge,
            "<=": operator.le,
            "!=": operator.ne,
            ">": operator.gt,
            "<": operator.lt,
            "==": operator.eq,
        }
        
        for op_str, op_func in operators_map.items():
            if op_str in expression:
                left, right = expression.split(op_str, 1)
                
                # Sustituir variable
                left = re.sub(rf'\b{variable}\b', str(value), left)
                right = re.sub(rf'\b{variable}\b', str(value), right)
                
                left_value = self.calculator.evaluate(left)
                right_value = self.calculator.evaluate(right)
                
                return op_func(left_value, right_value)
        
        raise ValueError(f"No se encontró operador de comparación en: {expression}")
    
    def check_answer_format(
        self,
        answer: str,
        expected_format: str = "numeric",
    ) -> tuple[bool, str | None]:
        """
        Verifica el formato de una respuesta.
        
        Args:
            answer: Respuesta a verificar.
            expected_format: Formato esperado (numeric, fraction, expression).
            
        Returns:
            Tupla (es_válido, mensaje).
        """
        answer = answer.strip()
        
        if expected_format == "numeric":
            try:
                float(answer)
                return True, None
            except ValueError:
                return False, "Se esperaba un valor numérico"
        
        elif expected_format == "fraction":
            # Formato a/b
            if re.match(r'^-?\d+/\d+$', answer):
                return True, None
            return False, "Se esperaba una fracción (ej: 3/4)"
        
        elif expected_format == "expression":
            valid, error = self.calculator.validate_expression(answer)
            return valid, error
        
        return True, None


# =============================================================================
# DETECTOR DE TIPO DE PROBLEMA
# =============================================================================

class ProblemClassifier:
    """
    Clasificador básico de tipos de problemas.
    
    Usa heurísticas simples para clasificar problemas por dominio.
    Para clasificación más precisa, usar el LLM.
    """
    
    # Palabras clave por dominio
    DOMAIN_KEYWORDS = {
        "mathematics": [
            "ecuación", "equation", "resolver", "solve", "calcular", "calculate",
            "derivada", "derivative", "integral", "límite", "limit",
            "función", "function", "polinomio", "polynomial", "matriz", "matrix",
            "factorial", "combinatoria", "probabilidad", "estadística",
            "área", "perímetro", "volumen", "triángulo", "círculo",
            "x", "y", "z", "=", "+", "-", "*", "/", "^",
            "raíz", "sqrt", "logaritmo", "log", "exponencial",
        ],
        "physics": [
            "velocidad", "velocity", "aceleración", "acceleration",
            "fuerza", "force", "masa", "mass", "energía", "energy",
            "trabajo", "work", "potencia", "power", "momentum",
            "gravedad", "gravity", "newton", "joule", "watt",
            "m/s", "km/h", "kg", "N", "J", "W",
            "cinemática", "dinámica", "termodinámica",
            "eléctrico", "magnético", "campo", "onda",
        ],
        "chemistry": [
            "mol", "moles", "gramos", "grams", "reacción", "reaction",
            "átomo", "atom", "molécula", "molecule", "elemento", "element",
            "pH", "ácido", "acid", "base", "sal", "salt",
            "oxidación", "reducción", "enlace", "bond",
            "estequiometría", "equilibrio químico",
            "H2O", "CO2", "NaCl", "→", "⟶",
        ],
        "programming": [
            "código", "code", "programa", "program", "función", "function",
            "algoritmo", "algorithm", "array", "lista", "list",
            "loop", "bucle", "for", "while", "if", "else",
            "python", "java", "javascript", "c++",
            "complejidad", "complexity", "O(n)", "recursión",
            "clase", "class", "objeto", "object", "método",
        ],
    }
    
    # Patrones regex por dominio
    DOMAIN_PATTERNS = {
        "mathematics": [
            r'\d+\s*[+\-*/^]\s*\d+',  # Operaciones numéricas
            r'[xyz]\s*=',              # Ecuaciones con variables
            r'\d*x\s*[+\-]\s*\d+\s*=', # Ecuaciones lineales
            r'f\s*\(x\)',              # Notación de función
        ],
        "physics": [
            r'\d+\s*(m/s|km/h|m/s²)',  # Unidades de velocidad/aceleración
            r'\d+\s*(kg|g|N|J|W)',      # Unidades físicas
            r'v\s*=\s*v0\s*\+',         # Ecuaciones cinemáticas
        ],
        "chemistry": [
            r'[A-Z][a-z]?\d*',          # Fórmulas químicas simples
            r'→|⟶|\+.*→',              # Reacciones químicas
            r'\d+\s*mol',               # Moles
        ],
    }
    
    def classify(self, problem_text: str) -> tuple[str, float]:
        """
        Clasifica un problema por dominio.
        
        Args:
            problem_text: Texto del problema.
            
        Returns:
            Tupla (dominio, confianza).
        """
        text_lower = problem_text.lower()
        scores: dict[str, float] = {}
        
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            
            # Bonus por patrones regex
            patterns = self.DOMAIN_PATTERNS.get(domain, [])
            for pattern in patterns:
                if re.search(pattern, problem_text, re.IGNORECASE):
                    score += 2
            
            scores[domain] = score
        
        if not scores or max(scores.values()) == 0:
            return "general", 0.5
        
        best_domain = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_domain] / total_score if total_score > 0 else 0.5
        
        return best_domain, min(confidence, 0.95)
    
    def estimate_difficulty(self, problem_text: str) -> str:
        """
        Estima la dificultad de un problema.
        
        Args:
            problem_text: Texto del problema.
            
        Returns:
            Nivel de dificultad (básico, intermedio, avanzado).
        """
        text_lower = problem_text.lower()
        
        # Indicadores de dificultad avanzada
        advanced_indicators = [
            "demuestra", "prove", "demostrar",
            "generaliza", "generalize",
            "múltiples variables", "sistema de ecuaciones",
            "integral", "derivada parcial", "ecuación diferencial",
            "optimización", "optimization",
            "inductión", "induction",
        ]
        
        # Indicadores de dificultad básica
        basic_indicators = [
            "simple", "básico", "basic",
            "suma", "resta", "multiplicación", "división",
            "¿cuánto es", "calcula",
        ]
        
        advanced_count = sum(1 for ind in advanced_indicators if ind in text_lower)
        basic_count = sum(1 for ind in basic_indicators if ind in text_lower)
        
        # También considerar longitud del problema
        word_count = len(problem_text.split())
        
        if advanced_count >= 2 or word_count > 100:
            return "avanzado"
        elif basic_count >= 2 or word_count < 20:
            return "básico"
        else:
            return "intermedio"


# =============================================================================
# INSTANCIAS GLOBALES
# =============================================================================

calculator = SafeCalculator()
validator = SolutionValidator()
classifier = ProblemClassifier()


__all__ = [
    "SafeCalculator",
    "SolutionValidator",
    "ProblemClassifier",
    "calculator",
    "validator",
    "classifier",
]
