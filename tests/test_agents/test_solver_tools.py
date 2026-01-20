"""
Tests para las herramientas del Solver (calculadora, validador, clasificador).
"""

from __future__ import annotations

import math

import pytest

from src.agents.solver.tools import (
    ProblemClassifier,
    SafeCalculator,
    SolutionValidator,
)


class TestSafeCalculator:
    """Tests para SafeCalculator."""
    
    @pytest.fixture
    def calc(self) -> SafeCalculator:
        return SafeCalculator()
    
    # Tests de operaciones básicas
    def test_addition(self, calc: SafeCalculator) -> None:
        assert calc.evaluate("2 + 3") == 5
    
    def test_subtraction(self, calc: SafeCalculator) -> None:
        assert calc.evaluate("10 - 4") == 6
    
    def test_multiplication(self, calc: SafeCalculator) -> None:
        assert calc.evaluate("3 * 4") == 12
    
    def test_division(self, calc: SafeCalculator) -> None:
        assert calc.evaluate("10 / 4") == 2.5
    
    def test_floor_division(self, calc: SafeCalculator) -> None:
        assert calc.evaluate("10 // 3") == 3
    
    def test_modulo(self, calc: SafeCalculator) -> None:
        assert calc.evaluate("10 % 3") == 1
    
    def test_power(self, calc: SafeCalculator) -> None:
        assert calc.evaluate("2 ** 3") == 8
    
    def test_negative(self, calc: SafeCalculator) -> None:
        assert calc.evaluate("-5") == -5
        assert calc.evaluate("--5") == 5
    
    # Tests de orden de operaciones
    def test_order_of_operations(self, calc: SafeCalculator) -> None:
        assert calc.evaluate("2 + 3 * 4") == 14
        assert calc.evaluate("(2 + 3) * 4") == 20
    
    def test_nested_parentheses(self, calc: SafeCalculator) -> None:
        assert calc.evaluate("((2 + 3) * (4 - 1))") == 15
    
    # Tests de funciones matemáticas
    def test_sqrt(self, calc: SafeCalculator) -> None:
        assert calc.evaluate("sqrt(16)") == 4.0
    
    def test_sin_cos(self, calc: SafeCalculator) -> None:
        assert abs(calc.evaluate("sin(0)")) < 1e-10
        assert abs(calc.evaluate("cos(0)") - 1) < 1e-10
    
    def test_log(self, calc: SafeCalculator) -> None:
        assert abs(calc.evaluate("log(e)") - 1) < 1e-10
    
    def test_abs(self, calc: SafeCalculator) -> None:
        assert calc.evaluate("abs(-5)") == 5
    
    def test_factorial(self, calc: SafeCalculator) -> None:
        assert calc.evaluate("factorial(5)") == 120
    
    # Tests de constantes
    def test_pi_constant(self, calc: SafeCalculator) -> None:
        result = calc.evaluate("pi")
        assert abs(result - math.pi) < 1e-10
    
    def test_e_constant(self, calc: SafeCalculator) -> None:
        result = calc.evaluate("e")
        assert abs(result - math.e) < 1e-10
    
    def test_expression_with_pi(self, calc: SafeCalculator) -> None:
        result = calc.evaluate("2 * pi")
        assert abs(result - 2 * math.pi) < 1e-10
    
    # Tests de seguridad
    def test_rejects_variable(self, calc: SafeCalculator) -> None:
        with pytest.raises(ValueError, match="Variable no definida"):
            calc.evaluate("x + 1")
    
    def test_rejects_unknown_function(self, calc: SafeCalculator) -> None:
        with pytest.raises(ValueError, match="Función no permitida"):
            calc.evaluate("eval('1+1')")
    
    def test_rejects_import(self, calc: SafeCalculator) -> None:
        with pytest.raises(ValueError):
            calc.evaluate("__import__('os')")
    
    # Tests de validación
    def test_validate_valid_expression(self, calc: SafeCalculator) -> None:
        is_valid, error = calc.validate_expression("2 + 2")
        assert is_valid is True
        assert error is None
    
    def test_validate_invalid_expression(self, calc: SafeCalculator) -> None:
        is_valid, error = calc.validate_expression("2 +")
        assert is_valid is False
        assert error is not None


class TestSolutionValidator:
    """Tests para SolutionValidator."""
    
    @pytest.fixture
    def validator(self) -> SolutionValidator:
        return SolutionValidator()
    
    def test_verify_equation_correct_solution(self, validator: SolutionValidator) -> None:
        is_valid, diff = validator.verify_equation_solution(
            equation="2*x + 3 = 7",
            variable="x",
            value=2,
        )
        assert is_valid is True
        assert diff < 1e-9
    
    def test_verify_equation_incorrect_solution(self, validator: SolutionValidator) -> None:
        is_valid, diff = validator.verify_equation_solution(
            equation="2*x + 3 = 7",
            variable="x",
            value=3,
        )
        assert is_valid is False
        assert diff == 2.0
    
    def test_verify_quadratic_solution(self, validator: SolutionValidator) -> None:
        # x^2 - 5x + 6 = 0 tiene soluciones x=2 y x=3
        is_valid_2, _ = validator.verify_equation_solution(
            equation="x**2 - 5*x + 6 = 0",
            variable="x",
            value=2,
        )
        is_valid_3, _ = validator.verify_equation_solution(
            equation="x**2 - 5*x + 6 = 0",
            variable="x",
            value=3,
        )
        assert is_valid_2 is True
        assert is_valid_3 is True
    
    def test_verify_inequality_true(self, validator: SolutionValidator) -> None:
        assert validator.verify_inequality("x > 0", "x", 5) is True
        assert validator.verify_inequality("x <= 10", "x", 10) is True
        assert validator.verify_inequality("2*x >= 4", "x", 3) is True
    
    def test_verify_inequality_false(self, validator: SolutionValidator) -> None:
        assert validator.verify_inequality("x > 0", "x", -1) is False
        assert validator.verify_inequality("x < 5", "x", 10) is False
    
    def test_check_numeric_format(self, validator: SolutionValidator) -> None:
        valid, _ = validator.check_answer_format("42", "numeric")
        assert valid is True
        
        valid, _ = validator.check_answer_format("3.14", "numeric")
        assert valid is True
        
        valid, _ = validator.check_answer_format("abc", "numeric")
        assert valid is False
    
    def test_check_fraction_format(self, validator: SolutionValidator) -> None:
        valid, _ = validator.check_answer_format("3/4", "fraction")
        assert valid is True
        
        valid, _ = validator.check_answer_format("-1/2", "fraction")
        assert valid is True
        
        valid, _ = validator.check_answer_format("0.5", "fraction")
        assert valid is False


class TestProblemClassifier:
    """Tests para ProblemClassifier."""
    
    @pytest.fixture
    def classifier(self) -> ProblemClassifier:
        return ProblemClassifier()
    
    def test_classify_math_equation(self, classifier: ProblemClassifier) -> None:
        domain, confidence = classifier.classify("Resuelve la ecuación 2x + 3 = 7")
        assert domain == "mathematics"
        assert confidence > 0.5
    
    def test_classify_math_calculus(self, classifier: ProblemClassifier) -> None:
        domain, _ = classifier.classify("Calcula la derivada de f(x) = x^2 + 3x")
        assert domain == "mathematics"
    
    def test_classify_physics_kinematics(self, classifier: ProblemClassifier) -> None:
        domain, _ = classifier.classify(
            "Un coche viaja a 60 km/h. ¿Qué distancia recorre en 2 horas?"
        )
        assert domain == "physics"
    
    def test_classify_physics_forces(self, classifier: ProblemClassifier) -> None:
        domain, _ = classifier.classify(
            "Calcula la fuerza necesaria para acelerar una masa de 5 kg a 2 m/s²"
        )
        assert domain == "physics"
    
    def test_classify_programming(self, classifier: ProblemClassifier) -> None:
        domain, _ = classifier.classify(
            "Escribe un algoritmo para ordenar una lista de números"
        )
        assert domain == "programming"
    
    def test_classify_chemistry(self, classifier: ProblemClassifier) -> None:
        domain, _ = classifier.classify(
            "Balancea la reacción: H2 + O2 → H2O"
        )
        assert domain == "chemistry"
    
    def test_classify_general_fallback(self, classifier: ProblemClassifier) -> None:
        domain, _ = classifier.classify("¿Cuál es tu color favorito?")
        # Debería ser general o tener baja confianza
        assert domain in ("general", "mathematics")  # Puede variar
    
    def test_estimate_difficulty_basic(self, classifier: ProblemClassifier) -> None:
        difficulty = classifier.estimate_difficulty("¿Cuánto es 2 + 2?")
        assert difficulty == "básico"
    
    def test_estimate_difficulty_advanced(self, classifier: ProblemClassifier) -> None:
        difficulty = classifier.estimate_difficulty(
            "Demuestra por inducción matemática que la suma de los primeros n "
            "números naturales es n(n+1)/2. Generaliza el resultado para "
            "progresiones aritméticas arbitrarias."
        )
        assert difficulty == "avanzado"
    
    def test_estimate_difficulty_intermediate(self, classifier: ProblemClassifier) -> None:
        difficulty = classifier.estimate_difficulty(
            "Resuelve el sistema de ecuaciones: 2x + y = 5, x - y = 1"
        )
        assert difficulty == "intermedio"
