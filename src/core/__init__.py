"""
Módulo core del sistema Aula AI Tutor.

Contiene tipos, estructuras de datos y excepciones fundamentales.
"""

from src.core.exceptions import (
    A2AConnectionError,
    A2AError,
    A2AProtocolError,
    A2ATimeoutError,
    AgentError,
    AulaError,
    BackendNotSupportedError,
    ConfigurationError,
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    GuardrailError,
    InsufficientVRAMError,
    InvalidModelIdError,
    JSONParsingError,
    ManipulationDetectedError,
    ModelConnectionError,
    ModelError,
    ModelGenerationError,
    ModelLoadError,
    ModelNotFoundError,
    ModelTimeoutError,
    ParsingError,
    PedagogicalValidationError,
    SolutionLeakDetectedError,
    SolutionNotAvailableError,
    SolutionParsingError,
    SolverError,
    TutorError,
)
from src.core.types import (
    A2AMessage,
    A2AMessageType,
    ConversationHistory,
    DifficultyLevel,
    EmbeddingResponse,
    GuardrailResult,
    Hint,
    HintLevel,
    Message,
    MessageRole,
    ModelResponse,
    ProblemType,
    SolutionStep,
    StructuredSolution,
    StudentInput,
    StudentIntent,
    TutorResponse,
    TutoringSession,
)

__all__ = [
    # Types - Enums
    "MessageRole",
    "ProblemType",
    "DifficultyLevel",
    "HintLevel",
    "StudentIntent",
    "GuardrailResult",
    "A2AMessageType",
    # Types - Messages
    "Message",
    "ConversationHistory",
    # Types - Solution
    "SolutionStep",
    "Hint",
    "StructuredSolution",
    # Types - Tutor
    "TutorResponse",
    "StudentInput",
    "TutoringSession",
    # Types - Model
    "ModelResponse",
    "EmbeddingResponse",
    # Types - A2A
    "A2AMessage",
    # Exceptions - Base
    "AulaError",
    # Exceptions - Model
    "ModelError",
    "ModelNotFoundError",
    "ModelConnectionError",
    "ModelGenerationError",
    "ModelTimeoutError",
    "ModelLoadError",
    "InsufficientVRAMError",
    # Exceptions - Configuration
    "ConfigurationError",
    "InvalidModelIdError",
    "BackendNotSupportedError",
    # Exceptions - Guardrails
    "GuardrailError",
    "ManipulationDetectedError",
    "SolutionLeakDetectedError",
    "PedagogicalValidationError",
    # Exceptions - Agents
    "AgentError",
    "SolverError",
    "TutorError",
    "SolutionNotAvailableError",
    # Exceptions - A2A
    "A2AError",
    "A2AConnectionError",
    "A2ATimeoutError",
    "A2AProtocolError",
    # Exceptions - Embedding
    "EmbeddingError",
    "EmbeddingDimensionMismatchError",
    # Exceptions - Parsing
    "ParsingError",
    "JSONParsingError",
    "SolutionParsingError",
]
