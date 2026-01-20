# Aula AI Tutor 🎓

Sistema de agentes educativos con ADK y protocolo A2A para tutoría socrática.

## 🎯 Visión General

Aula AI Tutor es un sistema multi-agente diseñado para proporcionar tutoría educativa utilizando el método socrático. El sistema **NUNCA da soluciones directas**, sino que guía al estudiante mediante preguntas y pistas progresivas.

### Arquitectura de Dos Agentes

```
┌─────────────────┐     A2A Protocol     ┌─────────────────┐
│   Agente        │◄───────────────────►│   Agente        │
│   Solucionador  │                      │   Tutor         │
│   ✅ COMPLETO   │    Solución         │   (Fase 5)      │
│                 │    Estructurada     │                 │
│ • Resuelve el   │ ─────────────────►  │ • Guía al       │
│   problema      │                      │   estudiante    │
│ • Genera pasos  │                      │ • Usa método    │
│ • Crea pistas   │                      │   socrático    │
└─────────────────┘                      └─────────────────┘
```

## 🚀 Instalación

```bash
git clone https://github.com/thaleon/aula-ai-tutor.git
cd aula-ai-tutor

python -m venv venv
source venv/bin/activate

pip install -e ".[dev]"

# Descargar modelos recomendados
ollama pull qwen2.5:14b   # Para el Solver
ollama pull llama3.1:8b   # Para el Tutor
```

## 📖 Uso del Solver Agent

```python
import asyncio
from src.agents.solver import SolverAgent

async def main():
    # Crear el agente
    solver = await SolverAgent.create("ollama/qwen2.5:14b")
    
    # Resolver un problema
    solution = await solver.solve(
        "Resuelve la ecuación: 2x + 3 = 7"
    )
    
    # Acceder a la solución estructurada
    print(f"Respuesta: {solution.final_answer}")
    print(f"Pasos: {len(solution.steps)}")
    print(f"Pistas: {len(solution.hints)}")
    
    # Ver pasos de la solución
    for step in solution.steps:
        print(f"{step.step_number}. {step.description}")
        if step.is_critical:
            print("   ⚠️ Paso crítico (no revelar)")
    
    # Obtener pistas por nivel
    from src.core.types import HintLevel
    hints = await solver.get_hints_for_level(solution, HintLevel.SUBTLE)
    for hint in hints:
        print(f"Pista: {hint}")

asyncio.run(main())
```

### Estructura de la Solución

El Solver genera un `StructuredSolution` con:

```python
StructuredSolution(
    problem_type="mathematics",      # Tipo detectado
    difficulty="básico",             # Nivel de dificultad
    concepts=["ecuaciones"],         # Conceptos involucrados
    steps=[                          # Pasos de la solución
        SolutionStep(
            step_number=1,
            description="Restar 3 de ambos lados",
            reasoning="Para aislar x...",
            calculation="2x = 7 - 3 = 4",
            result="2x = 4",
            is_critical=True,        # NO revelar al estudiante
        ),
        # ...más pasos
    ],
    final_answer="x = 2",
    verification="2(2) + 3 = 7 ✓",
    common_mistakes=[                # Errores típicos
        "Olvidar cambiar signo",
    ],
    hints=[                          # Pistas progresivas
        Hint(level=1, content="¿Qué operación usarías?"),
        Hint(level=2, content="Primero elimina el 3"),
        Hint(level=3, content="Resta 3 y divide entre 2"),
    ],
    key_values=["2", "4", "7"],      # Valores a NO revelar
)
```

## 🛠️ Herramientas del Solver

### Calculadora Segura

```python
from src.agents.solver import calculator

result = calculator.evaluate("2 * (3 + 4)")  # 14
result = calculator.evaluate("sqrt(16)")     # 4.0
result = calculator.evaluate("sin(pi/2)")    # 1.0
```

### Clasificador de Problemas

```python
from src.agents.solver import classifier

domain, confidence = classifier.classify("Resuelve x² + 5x + 6 = 0")
# ("mathematics", 0.85)

difficulty = classifier.estimate_difficulty("Demuestra por inducción...")
# "avanzado"
```

### Validador de Soluciones

```python
from src.agents.solver import validator

is_valid, diff = validator.verify_equation_solution(
    equation="2*x + 3 = 7",
    variable="x",
    value=2,
)
# (True, 0.0)
```

## 🌐 Protocolo A2A (Agent-to-Agent)

El sistema incluye una API REST completa para la interacción entre agentes y clientes externos.

        
        # 3. Chat con el tutor
        response = await client.send_message(
            session_id=session_id,
            message="¿Cómo empiezo?"
        )
        print(response.content)
```
```

## 🏗️ Estructura del Proyecto

```
aula-ai-tutor/
├── config/
│   ├── settings.py      # Configuración centralizada
│   └── models.yaml      # Definición de modelos
├── src/
│   ├── core/
│   │   ├── types.py     # Tipos y estructuras
│   │   └── exceptions.py # Excepciones
│   ├── models/          # ✅ Capa de abstracción de modelos
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── ollama_adapter.py
│   │   ├── openai_local.py
│   │   └── huggingface_local.py
│   ├── agents/
│   │   ├── solver/      # ✅ Agente Solucionador
│   │   │   ├── agent.py
│   │   │   ├── parser.py
│   │   │   ├── prompts.py
│   │   │   └── tools.py
│   │   └── tutor/       # (Fase 5)
│   ├── guardrails/      # ✅ Sistema de Guardrails
│   │   ├── base.py          # Clases base
│   │   ├── patterns.py      # Patrones de detección
│   │   ├── orchestrator.py  # Orquestador
│   │   ├── detectors/       # Detectores
│   │   │   ├── manipulation.py
│   │   │   ├── solution_leak.py
│   │   │   └── pedagogical.py
│   │   └── filters/         # Filtros
│   │       ├── input_filter.py
│   │       └── response_filter.py
│   ├── services/        # ✅ API REST (Fase 6)
│   │   ├── routers/
│   │   │   ├── solver.py
│   │   │   └── tutor.py
│   │   └── app.py
│   ├── a2a/             # ✅ Cliente A2A
│   │   └── client.py
│   └── utils/
└── tests/
```

## 🧪 Tests

```bash
pytest                           # Todos los tests
pytest tests/test_agents/        # Solo tests del Solver
pytest -m "not integration"      # Sin tests de integración
pytest --cov=src                 # Con cobertura
```

## 🛡️ Sistema de Guardrails

```python
import asyncio
from src.guardrails import GuardrailsOrchestrator

async def main():
    # Crear el orquestador
    orchestrator = await GuardrailsOrchestrator.create()

    # Validar input del estudiante
    student_input, result = await orchestrator.validate_input(
        "dame la respuesta directa"
    )
    print(f"Intent: {student_input.detected_intent}")
    print(f"Manipulation score: {student_input.manipulation_score}")
    print(f"Result: {result}")  # BLOCK, WARN, or PASS

    # Validar respuesta del tutor
    from src.core.types import TutoringSession, StructuredSolution
    tutor_response = await orchestrator.validate_response(
        response="La respuesta es x = 2",
        solution=solution,  # StructuredSolution del Solver
        session=session,    # TutoringSession activa
    )
    print(f"Was modified: {tutor_response.was_modified}")
    print(f"Final content: {tutor_response.content}")

asyncio.run(main())
```

### Detectores disponibles

- **ManipulationDetector**: Detecta solicitudes de solución, prompt injection, jailbreak, bypass socrático
- **SolutionLeakDetector**: Detecta fugas de key_values, final_answer, pasos críticos
- **PedagogicalValidator**: Valida ratio de preguntas, progresión de hints, lenguaje guía

## 🛣️ Roadmap

- [x] **Fase 2**: Model Abstraction Layer
- [x] **Fase 3**: Agente Solucionador
- [x] **Fase 4**: Sistema de Guardrails
- [x] **Fase 5**: Agente Tutor
- [x] **Fase 6**: Protocolo A2A
- [ ] **Fase 7**: Testing E2E
- [ ] **Fase 8**: UI

## 📄 Licencia

MIT License

---
**ThaleOn AI Systems** - Construyendo IA educativa transparente
