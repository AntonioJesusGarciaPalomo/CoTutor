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
ollama pull qwen2.5:7b    # Para el Solver (buena adherencia a JSON estructurado)
ollama pull llama3.2       # Para el Tutor
```

## ⚙️ Configuración

La configuración se gestiona mediante variables de entorno con prefijo `AULA_` o un archivo `.env`.

### Variables principales

| Variable | Default | Descripción |
|----------|---------|-------------|
| `AULA_MODEL_DEFAULTS__SOLVER_MODEL` | `ollama/qwen2.5:7b` | Modelo del Solver |
| `AULA_MODEL_DEFAULTS__TUTOR_MODEL` | `ollama/llama3.2:latest` | Modelo del Tutor |
| `AULA_MODEL_DEFAULTS__SOLVER_MAX_TOKENS` | `4096` | Tokens máximos para la respuesta JSON del Solver |
| `AULA_MODEL_DEFAULTS__TUTOR_MAX_TOKENS` | `1024` | Tokens máximos para respuestas del Tutor |
| `AULA_MODEL_DEFAULTS__SOLVER_TEMPERATURE` | `0.1` | Temperatura del Solver (determinista) |
| `AULA_MODEL_DEFAULTS__TUTOR_TEMPERATURE` | `0.7` | Temperatura del Tutor (creativo) |
| `AULA_OLLAMA__BASE_URL` | `http://localhost:11434` | URL del servidor Ollama |
| `AULA_OLLAMA__TIMEOUT` | `300` | Timeout de Ollama en segundos |
| `AULA_GUARDRAILS__MANIPULATION_DETECTION_ENABLED` | `true` | Activar detección de manipulación |
| `AULA_GUARDRAILS__SOLUTION_LEAK_DETECTION_ENABLED` | `true` | Activar detección de fugas |
| `AULA_PERFORMANCE__GENERATION_TIMEOUT` | `120` | Timeout general de generación |
| `AULA_LOGGING__LEVEL` | `INFO` | Nivel de logging |

### Backends soportados

El sistema soporta tres backends de modelos:

- **Ollama**: Servidor local de inferencia. Ideal para desarrollo. Soporta JSON mode nativo (`format: "json"`).
- **OpenAI-compatible** (vLLM, llama.cpp, LM Studio): Servidores que implementan la API de OpenAI. Soporta JSON mode via `response_format: {"type": "json_object"}`.
- **HuggingFace Transformers**: Carga directa de modelos con cuantización 4-bit/8-bit.

Los modelos se identifican con el formato `backend/model_name`:
```
ollama/qwen2.5:7b
openai_local/meta-llama/Llama-3.1-8B
huggingface/Qwen/Qwen2.5-14B-Instruct
```

## 📖 Uso del Solver Agent

```python
import asyncio
from src.agents.solver import SolverAgent

async def main():
    # Crear el agente (max_tokens y temperature son configurables)
    solver = await SolverAgent.create(
        "ollama/qwen2.5:7b",
        max_tokens=4096,
        temperature=0.1,
    )

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

### Robustez del Solver (JSON parsing)

El Solver utiliza un sistema de parsing de 7 capas para garantizar que siempre se obtiene una solución válida, incluso con modelos pequeños:

1. **JSON Mode nativo**: Los backends Ollama y OpenAI-compatible fuerzan JSON válido desde el motor de inferencia (Constrained Decoding).
2. **json-repair**: Librería que repara automáticamente JSON truncado, comillas sin cerrar, trailing commas y JSON dentro de markdown.
3. **Extracción manual + json.loads**: Busca JSON en bloques de código, texto mixto, etc.
4. **Extracción + json_repair**: Combina extracción manual con reparación automática.
5. **Reparación de truncado**: Análisis de estado que detecta y cierra strings, brackets y arrays abiertos.
6. **Prompt simplificado**: Template JSON fill-in-the-blank como fallback para modelos pequeños.
7. **Solución mínima**: Genera una estructura válida con la información disponible.

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

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Health check del servicio |
| `POST` | `/solver/solve` | Resuelve un problema y devuelve `StructuredSolution` |
| `POST` | `/tutor/session` | Inicia una sesión de tutoría |
| `POST` | `/tutor/chat` | Envía un mensaje al tutor en una sesión activa |

### Ejemplo de uso

```python
import httpx

# 1. Resolver problema
response = httpx.post("http://localhost:8000/solver/solve", json={
    "problem_text": "Resuelve 2x + 3 = 7",
    "domain_hint": "mathematics",
})
solution = response.json()

# 2. Iniciar sesión de tutoría
response = httpx.post("http://localhost:8000/tutor/session", json={
    "problem_text": "Resuelve 2x + 3 = 7",
    "solution": solution,
})
session_id = response.json()["session_id"]

# 3. Chat con el tutor
response = httpx.post("http://localhost:8000/tutor/chat", json={
    "session_id": session_id,
    "message": "¿Cómo empiezo?",
})
print(response.json()["content"])
```

## 🏗️ Estructura del Proyecto

```
aula-ai-tutor/
├── config/
│   ├── settings.py      # Configuración centralizada (Pydantic Settings)
│   └── models.yaml      # Definición de modelos
├── src/
│   ├── core/
│   │   ├── types.py     # Tipos y estructuras (Pydantic models)
│   │   └── exceptions.py # Jerarquía de excepciones
│   ├── models/          # Capa de abstracción de modelos
│   │   ├── base.py          # BaseModelAdapter (interfaz abstracta)
│   │   ├── factory.py       # ModelFactory (singleton caching)
│   │   ├── ollama_adapter.py    # Backend Ollama (con JSON mode)
│   │   ├── openai_local.py      # Backend OpenAI-compatible (con JSON mode)
│   │   └── huggingface_local.py # Backend HuggingFace (4-bit/8-bit)
│   ├── agents/
│   │   ├── solver/      # Agente Solucionador
│   │   │   ├── agent.py     # Lógica principal + retry con prompt simplificado
│   │   │   ├── parser.py    # Parser JSON robusto (json-repair + fallbacks)
│   │   │   ├── prompts.py   # Prompts optimizados por dominio
│   │   │   └── tools.py     # SafeCalculator
│   │   └── tutor/       # Agente Tutor
│   │       ├── agent.py     # Lógica socrática
│   │       ├── prompts.py   # Prompts del tutor
│   │       ├── session_manager.py # Persistencia de sesiones
│   │       └── strategies.py     # Estrategias de enseñanza
│   ├── guardrails/      # Sistema de Guardrails
│   │   ├── base.py          # Clases base
│   │   ├── orchestrator.py  # Orquestador (pipeline)
│   │   ├── detectors/       # Detectores
│   │   │   ├── manipulation.py    # Prompt injection, jailbreak
│   │   │   ├── solution_leak.py   # Fuga de respuestas/key values
│   │   │   └── pedagogical.py     # Validación socrática
│   │   └── filters/         # Filtros
│   │       ├── input_filter.py    # Sanitización de entrada
│   │       └── response_filter.py # Filtrado de respuesta
│   ├── services/        # API REST (FastAPI)
│   │   ├── app.py           # Aplicación FastAPI
│   │   └── routers/
│   │       ├── solver.py    # Endpoints del Solver
│   │       └── tutor.py     # Endpoints del Tutor
│   ├── a2a/             # Cliente A2A
│   │   └── client.py
│   └── utils/
│       ├── logging.py   # Logging estructurado (structlog)
│       └── metrics.py   # Métricas de rendimiento
├── ui/                  # Frontend React + TypeScript
├── tests/               # Tests (pytest + pytest-asyncio)
└── pyproject.toml       # Dependencias y configuración del proyecto
```

## 🧪 Tests

```bash
pytest                           # Todos los tests
pytest tests/test_agents/        # Solo tests del Solver
pytest -m "not integration"      # Sin tests de integración
pytest --cov=src                 # Con cobertura
```

## 🛡️ Sistema de Guardrails

El sistema implementa una defensa de tres capas para proteger la integridad pedagógica:

### Validación de entrada (estudiante)

1. **InputFilter**: Sanitización y normalización del texto
2. **ManipulationDetector**: Detecta prompt injection, jailbreak y bypass socrático
3. **Clasificación de intención**: 6 tipos de intención del estudiante

### Validación de salida (tutor)

1. **SolutionLeakDetector**: Detecta fuga de key_values, final_answer y pasos críticos
2. **PedagogicalValidator**: Valida ratio de preguntas (mín. 30%), progresión de hints
3. **ResponseFilter**: Modifica o bloquea contenido no pedagógico

### Uso

```python
import asyncio
from src.guardrails import GuardrailsOrchestrator

async def main():
    orchestrator = await GuardrailsOrchestrator.create()

    # Validar input del estudiante
    student_input, result = await orchestrator.validate_input(
        "dame la respuesta directa"
    )
    print(f"Intent: {student_input.detected_intent}")
    print(f"Manipulation score: {student_input.manipulation_score}")
    print(f"Result: {result}")  # BLOCK, WARN, or PASS

    # Validar respuesta del tutor
    tutor_response = await orchestrator.validate_response(
        response="La respuesta es x = 2",
        solution=solution,  # StructuredSolution del Solver
        session=session,    # TutoringSession activa
    )
    print(f"Was modified: {tutor_response.was_modified}")
    print(f"Final content: {tutor_response.content}")

asyncio.run(main())
```

## 🛣️ Roadmap

- [x] **Fase 2**: Model Abstraction Layer (Ollama, OpenAI-compatible, HuggingFace)
- [x] **Fase 3**: Agente Solucionador (con JSON mode y parsing robusto)
- [x] **Fase 4**: Sistema de Guardrails
- [x] **Fase 5**: Agente Tutor
- [x] **Fase 6**: Protocolo A2A
- [ ] **Fase 7**: Testing E2E
- [ ] **Fase 8**: UI

## 📄 Licencia

MIT License

---
**ThaleOn AI Systems** - Construyendo IA educativa transparente
