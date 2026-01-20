"""
Script para ejecutar tests E2E.

Uso:
    python scripts/run_e2e.py                   # Ejecuta tests con mocks (rápido)
    python scripts/run_e2e.py --real            # Ejecuta contra modelos por defecto
    python scripts/run_e2e.py --real --solver-model ollama/codellama:7b --tutor-model ollama/codellama:7b
"""

import argparse
import sys
import os

from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Ejecutar tests E2E")
    parser.add_argument("--real", action="store_true", help="Usar modelos reales en lugar de mocks")
    parser.add_argument("--solver-model", help="Sobrescribir modelo del Solver (ej: ollama/codellama:7b)")
    parser.add_argument("--tutor-model", help="Sobrescribir modelo del Tutor (ej: ollama/llama3.1:8b)")
    args = parser.parse_args()
    
    # Directorio base
    root_dir = Path(__file__).parent.parent
    test_dir = root_dir / "tests" / "e2e"
    
    pytest_args = [
        str(test_dir),
        "-v",
        "--tb=short", # Traceback más limpio
    ]
    
    if args.real:
        print("🚀 MODO REAL: Ejecutando tests contra modelos LLM reales...")
        os.environ["E2E_REAL_MODELS"] = "true"
        
        if args.solver_model:
            print(f"   Solver Model: {args.solver_model}")
            os.environ["AULA_MODEL_DEFAULTS__SOLVER_MODEL"] = args.solver_model
        
        if args.tutor_model:
            print(f"   Tutor Model: {args.tutor_model}")
            os.environ["AULA_MODEL_DEFAULTS__TUTOR_MODEL"] = args.tutor_model
            
    else:
        print("⚡ MODO MOCK: Ejecutando tests con respuestas simuladas...")
        if "E2E_REAL_MODELS" in os.environ:
            del os.environ["E2E_REAL_MODELS"]
    
    # Ejecutar pytest usando el mismo intérprete de Python
    import subprocess
    
    cmd = [sys.executable, "-m", "pytest"] + pytest_args
    print(f"Ejecutando: {' '.join(cmd)}")
    
    try:
        result = subprocess.call(cmd)
        sys.exit(result)
    except Exception as e:
        print(f"Error al ejecutar pytest: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
