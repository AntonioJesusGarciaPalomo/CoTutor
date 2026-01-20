"""
Tipos y estructuras de datos del sistema Aula AI Tutor.

Este módulo define los tipos centrales usados en todo el sistema,
incluyendo mensajes, respuestas, soluciones estructuradas, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Enumeraciones
# =============================================================================

class MessageRole(str, Enum):
    """Roles en una conversación."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ProblemType(str, Enum):
    """Tipos de problemas soportados."""
    MATHEMATICS = "mathematics"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    PROGRAMMING = "programming"
    LOGIC = "logic"
    GENERAL = "general"


class DifficultyLevel(str, Enum):
    """Niveles de dificultad."""
    BASIC = "básico"
    INTERMEDIATE = "intermedio"
    ADVANCED = "avanzado"


class HintLevel(int, Enum):
    """Niveles de pistas (de más sutil a más directa)."""
    SUBTLE = 1
    MODERATE = 2
    DIRECT = 3


class StudentIntent(str, Enum):
    """Intención detectada del estudiante."""
    LEGITIMATE_QUESTION = "legitimate_question"
    HINT_REQUEST = "hint_request"
    VERIFICATION_REQUEST = "verification_request"
    SOLUTION_ATTEMPT = "solution_attempt"
    MANIPULATION_ATTEMPT = "manipulation_attempt"
    OFF_TOPIC = "off_topic"
    GREETING = "greeting"
    CLARIFICATION = "clarification"


class GuardrailResult(str, Enum):
    """Resultado de un guardrail."""
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


# =============================================================================
# Mensajes
# =============================================================================

class Message(BaseModel):
    """Mensaje en una conversación."""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class ConversationHistory(BaseModel):
    """Historial de conversación."""
    messages: list[Message] = Field(default_factory=list)
    session_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, role: MessageRole, content: str, **metadata: Any) -> None:
        """Añade un mensaje al historial."""
        self.messages.append(Message(role=role, content=content, metadata=metadata))
    
    def get_last_n(self, n: int) -> list[Message]:
        """Obtiene los últimos n mensajes."""
        return self.messages[-n:] if n > 0 else []
    
    def to_openai_format(self) -> list[dict[str, str]]:
        """Convierte a formato OpenAI para APIs."""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]


# =============================================================================
# Solución Estructurada (Output del Solver)
# =============================================================================

class SolutionStep(BaseModel):
    """Un paso en la solución de un problema."""
    step_number: int
    description: str
    reasoning: str
    calculation: str | None = None
    result: str | None = None
    is_critical: bool = False  # Marca pasos que NO deben revelarse


class Hint(BaseModel):
    """Una pista para el estudiante."""
    level: HintLevel
    content: str
    concepts_referenced: list[str] = Field(default_factory=list)


class StructuredSolution(BaseModel):
    """
    Solución estructurada generada por el Solver.
    
    Esta estructura es CONFIDENCIAL y solo el Tutor debe tener acceso.
    Nunca debe ser expuesta directamente al estudiante.
    """
    # Metadatos del problema
    problem_id: UUID = Field(default_factory=uuid4)
    problem_text: str
    problem_type: ProblemType
    difficulty: DifficultyLevel
    
    # Conceptos y prerrequisitos
    concepts: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    
    # Solución paso a paso
    steps: list[SolutionStep] = Field(default_factory=list)
    final_answer: str
    verification: str | None = None
    
    # Material de apoyo para el Tutor
    common_mistakes: list[str] = Field(default_factory=list)
    hints: list[Hint] = Field(default_factory=list)
    theory_references: list[str] = Field(default_factory=list)
    
    # Datos numéricos/clave que NO deben revelarse
    key_values: list[str] = Field(default_factory=list)
    
    # Metadatos
    created_at: datetime = Field(default_factory=datetime.now)
    solver_model: str | None = None
    confidence_score: float | None = None
    
    def get_hints_for_level(self, level: HintLevel) -> list[Hint]:
        """Obtiene pistas hasta cierto nivel (inclusive)."""
        return [h for h in self.hints if h.level.value <= level.value]
    
    def get_critical_steps(self) -> list[SolutionStep]:
        """Obtiene los pasos críticos que no deben revelarse."""
        return [s for s in self.steps if s.is_critical]


# =============================================================================
# Respuesta del Tutor
# =============================================================================

class TutorResponse(BaseModel):
    """Respuesta generada por el Tutor."""
    content: str
    
    # Análisis de la respuesta
    contains_question: bool = False
    hint_level_used: HintLevel | None = None
    strategy_used: str | None = None  # socratic, hint, redirection, etc.
    
    # Resultado de guardrails
    guardrail_results: dict[str, GuardrailResult] = Field(default_factory=dict)
    was_modified: bool = False
    original_content: str | None = None
    
    # Metadatos
    response_time_ms: float | None = None
    tutor_model: str | None = None


# =============================================================================
# Entrada del Estudiante
# =============================================================================

class StudentInput(BaseModel):
    """Entrada procesada del estudiante."""
    raw_content: str
    processed_content: str
    
    # Análisis
    detected_intent: StudentIntent
    intent_confidence: float = 0.0
    
    # Guardrails aplicados
    manipulation_score: float = 0.0
    is_on_topic: bool = True
    
    # Metadatos
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Sesión de Tutoría
# =============================================================================

class TutoringSession(BaseModel):
    """Sesión completa de tutoría para un problema."""
    session_id: UUID = Field(default_factory=uuid4)
    
    # Problema y solución
    problem_text: str
    solution: StructuredSolution | None = None
    
    # Historial
    conversation: ConversationHistory = Field(default_factory=ConversationHistory)
    
    # Estado
    current_hint_level: HintLevel = HintLevel.SUBTLE
    hints_given: int = 0
    questions_asked: int = 0
    
    # Métricas
    started_at: datetime = Field(default_factory=datetime.now)
    ended_at: datetime | None = None
    student_reached_solution: bool = False
    
    def advance_hint_level(self) -> HintLevel:
        """Avanza al siguiente nivel de pista si es posible."""
        if self.current_hint_level == HintLevel.SUBTLE:
            self.current_hint_level = HintLevel.MODERATE
        elif self.current_hint_level == HintLevel.MODERATE:
            self.current_hint_level = HintLevel.DIRECT
        self.hints_given += 1
        return self.current_hint_level


# =============================================================================
# Respuesta del Modelo (genérica)
# =============================================================================

class ModelResponse(BaseModel):
    """Respuesta genérica de un modelo de lenguaje."""
    content: str
    model: str
    
    # Métricas de generación
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    
    # Timing
    generation_time_ms: float | None = None
    tokens_per_second: float | None = None
    
    # Información adicional del backend
    finish_reason: str | None = None
    raw_response: dict[str, Any] | None = None


class EmbeddingResponse(BaseModel):
    """Respuesta de un modelo de embeddings."""
    embeddings: list[list[float]]
    model: str
    dimensions: int
    
    # Métricas
    total_tokens: int | None = None
    generation_time_ms: float | None = None


# =============================================================================
# Tipos para A2A Protocol
# =============================================================================

class A2AMessageType(str, Enum):
    """Tipos de mensajes en el protocolo A2A."""
    SOLVE_REQUEST = "solve_request"
    SOLVE_RESPONSE = "solve_response"
    HEALTH_CHECK = "health_check"
    ERROR = "error"


class A2AMessage(BaseModel):
    """Mensaje en el protocolo A2A."""
    message_id: UUID = Field(default_factory=uuid4)
    message_type: A2AMessageType
    payload: dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    sender: str
    recipient: str


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "MessageRole",
    "ProblemType",
    "DifficultyLevel",
    "HintLevel",
    "StudentIntent",
    "GuardrailResult",
    "A2AMessageType",
    # Messages
    "Message",
    "ConversationHistory",
    # Solution
    "SolutionStep",
    "Hint",
    "StructuredSolution",
    # Tutor
    "TutorResponse",
    "StudentInput",
    "TutoringSession",
    # Model
    "ModelResponse",
    "EmbeddingResponse",
    # A2A
    "A2AMessage",
]
