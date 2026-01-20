#!/usr/bin/env python3
"""
Ejemplo de uso del Agente Solucionador.

Este script demuestra cómo usar el SolverAgent para resolver
problemas educativos y obtener soluciones estructuradas.

Requisitos:
- Ollama corriendo en localhost:11434
- Modelo qwen2.5:14b descargado (o ajustar el modelo en el código)

Ejecutar:
    python examples/solver_example.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.solver import SolverAgent, classifier, calculator
from src.core.types import HintLevel


async def basic_solver_example():
    """Ejemplo básico del Solver."""
    print("\n" + "=" * 70)
    print("EJEMPLO 1: Resolución Básica de Ecuación")
    print("=" * 70)
    
    try:
        # Crear el agente (usa modelo por defecto o ajusta aquí)
        print("\n⏳ Creando SolverAgent...")
        solver = await SolverAgent.create("ollama/llama3.2:1b")  # Modelo pequeño para demo
        
        # Problema a resolver
        problem = "Resuelve la ecuación: 2x + 5 = 13"
        
        print(f"\n📝 Problema: {problem}")
        print("\n⏳ Resolviendo...")
        
        # Resolver el problema
        solution = await solver.solve(problem)
        
        # Mostrar resultados
        print(f"\n✅ Solución encontrada!")
        print(f"\n📊 Tipo de problema: {solution.problem_type.value}")
        print(f"📊 Dificultad: {solution.difficulty.value}")
        print(f"📊 Conceptos: {', '.join(solution.concepts)}")
        
        print(f"\n📋 PASOS DE LA SOLUCIÓN ({len(solution.steps)} pasos):")
        for step in solution.steps:
            critical = " ⚠️ [CRÍTICO]" if step.is_critical else ""
            print(f"\n   Paso {step.step_number}{critical}:")
            print(f"   └─ {step.description}")
            if step.calculation:
                print(f"   └─ Cálculo: {step.calculation}")
            if step.result:
                print(f"   └─ Resultado: {step.result}")
        
        print(f"\n🎯 RESPUESTA FINAL: {solution.final_answer}")
        
        if solution.verification:
            print(f"\n✓ Verificación: {solution.verification}")
        
        print(f"\n💡 PISTAS PARA EL TUTOR ({len(solution.hints)}):")
        for hint in solution.hints:
            level_emoji = {1: "🔵", 2: "🟡", 3: "🔴"}
            print(f"   {level_emoji.get(hint.level.value, '•')} Nivel {hint.level.value}: {hint.content}")
        
        if solution.common_mistakes:
            print(f"\n⚠️ ERRORES COMUNES:")
            for mistake in solution.common_mistakes:
                print(f"   • {mistake}")
        
        if solution.key_values:
            print(f"\n🔒 VALORES CLAVE (no revelar): {', '.join(solution.key_values)}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def physics_problem_example():
    """Ejemplo con problema de física."""
    print("\n" + "=" * 70)
    print("EJEMPLO 2: Problema de Física")
    print("=" * 70)
    
    try:
        solver = await SolverAgent.create("ollama/llama3.2:1b")
        
        problem = """Un coche parte del reposo y acelera uniformemente a 2 m/s². 
        ¿Qué distancia habrá recorrido después de 10 segundos?"""
        
        print(f"\n📝 Problema: {problem}")
        print("\n⏳ Resolviendo...")
        
        # Indicar dominio para mejor resultado
        solution = await solver.solve(problem, domain_hint="physics")
        
        print(f"\n✅ Tipo: {solution.problem_type.value}")
        print(f"🎯 Respuesta: {solution.final_answer}")
        
        print(f"\n📋 Resumen de pasos:")
        for step in solution.steps:
            print(f"   {step.step_number}. {step.description}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


async def programming_problem_example():
    """Ejemplo con problema de programación."""
    print("\n" + "=" * 70)
    print("EJEMPLO 3: Problema de Programación")
    print("=" * 70)
    
    try:
        solver = await SolverAgent.create("ollama/llama3.2:1b")
        
        problem = """Escribe un algoritmo para encontrar el número más grande 
        en una lista de números."""
        
        print(f"\n📝 Problema: {problem}")
        print("\n⏳ Resolviendo...")
        
        solution = await solver.solve(problem, domain_hint="programming")
        
        print(f"\n✅ Tipo: {solution.problem_type.value}")
        print(f"🎯 Respuesta:\n{solution.final_answer}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


async def tools_demo():
    """Demostración de las herramientas auxiliares."""
    print("\n" + "=" * 70)
    print("EJEMPLO 4: Herramientas Auxiliares")
    print("=" * 70)
    
    # Calculadora segura
    print("\n🔢 CALCULADORA SEGURA:")
    expressions = [
        "2 + 3 * 4",
        "(2 + 3) * 4",
        "sqrt(16) + 2**3",
        "sin(pi/2)",
        "log(e)",
    ]
    
    for expr in expressions:
        try:
            result = calculator.evaluate(expr)
            print(f"   {expr} = {result}")
        except Exception as e:
            print(f"   {expr} → Error: {e}")
    
    # Clasificador de problemas
    print("\n🏷️ CLASIFICADOR DE PROBLEMAS:")
    problems = [
        "Resuelve x² + 5x + 6 = 0",
        "Un tren viaja a 80 km/h...",
        "Escribe una función en Python...",
        "Balancea: H2 + O2 → H2O",
    ]
    
    for prob in problems:
        domain, confidence = classifier.classify(prob)
        difficulty = classifier.estimate_difficulty(prob)
        print(f"   \"{prob[:30]}...\"")
        print(f"      └─ Dominio: {domain} ({confidence:.0%}), Dificultad: {difficulty}")


async def cache_demo():
    """Demostración del sistema de caché."""
    print("\n" + "=" * 70)
    print("EJEMPLO 5: Sistema de Caché")
    print("=" * 70)
    
    try:
        solver = await SolverAgent.create("ollama/llama3.2:1b")
        problem = "¿Cuánto es 5 + 3?"
        
        print(f"\n📝 Problema: {problem}")
        
        # Primera resolución
        print("\n⏳ Primera resolución (generando)...")
        import time
        start = time.time()
        solution1 = await solver.solve(problem)
        time1 = time.time() - start
        print(f"   ⏱️ Tiempo: {time1:.2f}s")
        
        # Segunda resolución (desde caché)
        print("\n⏳ Segunda resolución (desde caché)...")
        start = time.time()
        solution2 = await solver.solve(problem)
        time2 = time.time() - start
        print(f"   ⏱️ Tiempo: {time2:.4f}s")
        
        print(f"\n📊 Speedup por caché: {time1/time2:.1f}x más rápido")
        
        # Estadísticas de caché
        stats = solver.get_cache_stats()
        print(f"\n📈 Estadísticas de caché:")
        print(f"   • Tamaño actual: {stats['size']}")
        print(f"   • Tamaño máximo: {stats['max_size']}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


async def hints_progression_example():
    """Demostración de pistas progresivas."""
    print("\n" + "=" * 70)
    print("EJEMPLO 6: Pistas Progresivas")
    print("=" * 70)
    
    try:
        solver = await SolverAgent.create("ollama/llama3.2:1b")
        
        problem = "Resuelve: 3x - 7 = 2x + 5"
        print(f"\n📝 Problema: {problem}")
        
        solution = await solver.solve(problem)
        
        print("\n💡 Simulación de sesión de tutoría:")
        print("   (El tutor va revelando pistas según el estudiante lo necesite)\n")
        
        for level in [HintLevel.SUBTLE, HintLevel.MODERATE, HintLevel.DIRECT]:
            hints = await solver.get_hints_for_level(solution, level)
            level_name = {
                HintLevel.SUBTLE: "Nivel 1 (Sutil)",
                HintLevel.MODERATE: "Nivel 2 (Moderado)",
                HintLevel.DIRECT: "Nivel 3 (Directo)",
            }
            print(f"   📌 {level_name[level]}:")
            for hint in hints:
                print(f"      → {hint}")
            print()
        
        print(f"   🎯 (Respuesta que el tutor NUNCA debe revelar: {solution.final_answer})")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


async def main():
    """Ejecuta todos los ejemplos."""
    print("🎓 Aula AI Tutor - Ejemplos del Agente Solucionador")
    print("=" * 70)
    
    # Primero verificar que Ollama está disponible
    from src.models import ModelFactory
    model = ModelFactory.create("ollama/llama3.2:1b")
    is_healthy = await model.health_check()
    
    if not is_healthy:
        print("\n❌ Ollama no está disponible.")
        print("   Por favor, asegúrate de que Ollama está corriendo:")
        print("   $ ollama serve")
        print("   $ ollama pull llama3.2:1b")
        return
    
    print("✅ Ollama está disponible")
    
    # Ejecutar ejemplos
    await basic_solver_example()
    await physics_problem_example()
    await tools_demo()
    await cache_demo()
    await hints_progression_example()
    
    print("\n" + "=" * 70)
    print("✅ Ejemplos completados")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
