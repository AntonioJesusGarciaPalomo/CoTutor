"""
Excepciones personalizadas del sistema Aula AI Tutor.

Este módulo define una jerarquía de excepciones que permite
un manejo de errores preciso y consistente en todo el sistema.
"""

from __future__ import annotations

from typing import Any


# =============================================================================
# Excepción Base
# =============================================================================

class AulaError(Exception):
    """
    Excepción base para todos los errores del sistema Aula AI Tutor.
    
    Attributes:
        message: Mensaje descriptivo del error.
        details: Información adicional sobre el error.
        recoverable: Indica si el error es recuperable.
    """
    
    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        recoverable: bool = True,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.recoverable = recoverable
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message
    
    def to_dict(self) -> dict[str, Any]:
        """Convierte la excepción a un diccionario serializable."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
        }


# =============================================================================
# Errores de Modelo
# =============================================================================

class ModelError(AulaError):
    """Error relacionado con modelos de lenguaje."""
    pass


class ModelNotFoundError(ModelError):
    """El modelo solicitado no fue encontrado."""
    
    def __init__(self, model_id: str, backend: str | None = None) -> None:
        details = {"model_id": model_id}
        if backend:
            details["backend"] = backend
        super().__init__(
            f"Modelo no encontrado: {model_id}",
            details=details,
            recoverable=False,
        )


class ModelConnectionError(ModelError):
    """Error de conexión con el backend del modelo."""
    
    def __init__(self, backend: str, base_url: str, cause: str | None = None) -> None:
        details = {"backend": backend, "base_url": base_url}
        if cause:
            details["cause"] = cause
        super().__init__(
            f"No se pudo conectar al backend {backend} en {base_url}",
            details=details,
            recoverable=True,
        )


class ModelGenerationError(ModelError):
    """Error durante la generación de texto."""
    
    def __init__(self, model: str, cause: str, prompt_preview: str | None = None) -> None:
        details = {"model": model, "cause": cause}
        if prompt_preview:
            details["prompt_preview"] = prompt_preview[:200] + "..."
        super().__init__(
            f"Error generando respuesta con {model}: {cause}",
            details=details,
            recoverable=True,
        )


class ModelTimeoutError(ModelError):
    """Timeout durante la generación."""
    
    def __init__(self, model: str, timeout_seconds: float) -> None:
        super().__init__(
            f"Timeout de {timeout_seconds}s excedido para modelo {model}",
            details={"model": model, "timeout_seconds": timeout_seconds},
            recoverable=True,
        )


class ModelLoadError(ModelError):
    """Error al cargar un modelo (HuggingFace local)."""
    
    def __init__(self, model_id: str, cause: str) -> None:
        super().__init__(
            f"Error cargando modelo {model_id}: {cause}",
            details={"model_id": model_id, "cause": cause},
            recoverable=False,
        )


class InsufficientVRAMError(ModelError):
    """VRAM insuficiente para cargar el modelo."""
    
    def __init__(
        self,
        model_id: str,
        required_vram: str,
        available_vram: str | None = None,
    ) -> None:
        details = {"model_id": model_id, "required_vram": required_vram}
        if available_vram:
            details["available_vram"] = available_vram
        super().__init__(
            f"VRAM insuficiente para {model_id}. Requerido: {required_vram}",
            details=details,
            recoverable=False,
        )


# =============================================================================
# Errores de Configuración
# =============================================================================

class ConfigurationError(AulaError):
    """Error de configuración del sistema."""
    
    def __init__(self, message: str, config_key: str | None = None) -> None:
        details = {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, details=details, recoverable=False)


class InvalidModelIdError(ConfigurationError):
    """Formato de model_id inválido."""
    
    def __init__(self, model_id: str) -> None:
        super().__init__(
            f"Formato de model_id inválido: '{model_id}'. "
            f"Use el formato 'backend/model_name' (ej: 'ollama/llama3.1:8b')",
            config_key="model_id",
        )


class BackendNotSupportedError(ConfigurationError):
    """Backend no soportado."""
    
    def __init__(self, backend: str, supported_backends: list[str]) -> None:
        super().__init__(
            f"Backend '{backend}' no soportado. "
            f"Backends disponibles: {', '.join(supported_backends)}",
            config_key="backend",
        )


# =============================================================================
# Errores de Guardrails
# =============================================================================

class GuardrailError(AulaError):
    """Error en el sistema de guardrails."""
    pass


class ManipulationDetectedError(GuardrailError):
    """Se detectó un intento de manipulación."""
    
    def __init__(
        self,
        input_text: str,
        manipulation_type: str,
        confidence: float,
    ) -> None:
        super().__init__(
            f"Intento de manipulación detectado: {manipulation_type}",
            details={
                "manipulation_type": manipulation_type,
                "confidence": confidence,
                "input_preview": input_text[:100] + "..." if len(input_text) > 100 else input_text,
            },
            recoverable=True,
        )


class SolutionLeakDetectedError(GuardrailError):
    """Se detectó una fuga de solución en la respuesta."""
    
    def __init__(
        self,
        response_preview: str,
        leak_type: str,
        similarity_score: float | None = None,
    ) -> None:
        details = {
            "leak_type": leak_type,
            "response_preview": response_preview[:100] + "...",
        }
        if similarity_score is not None:
            details["similarity_score"] = similarity_score
        super().__init__(
            f"Fuga de solución detectada: {leak_type}",
            details=details,
            recoverable=True,
        )


class PedagogicalValidationError(GuardrailError):
    """La respuesta no cumple los criterios pedagógicos."""
    
    def __init__(self, reason: str, metrics: dict[str, Any] | None = None) -> None:
        super().__init__(
            f"Validación pedagógica fallida: {reason}",
            details={"reason": reason, "metrics": metrics or {}},
            recoverable=True,
        )


# =============================================================================
# Errores de Agentes
# =============================================================================

class AgentError(AulaError):
    """Error relacionado con agentes."""
    pass


class SolverError(AgentError):
    """Error en el agente Solver."""
    
    def __init__(self, message: str, problem_preview: str | None = None) -> None:
        details = {}
        if problem_preview:
            details["problem_preview"] = problem_preview[:200]
        super().__init__(message, details=details)


class TutorError(AgentError):
    """Error en el agente Tutor."""
    
    def __init__(self, message: str, session_id: str | None = None) -> None:
        details = {}
        if session_id:
            details["session_id"] = session_id
        super().__init__(message, details=details)


class SolutionNotAvailableError(AgentError):
    """El Tutor no tiene acceso a la solución."""
    
    def __init__(self, problem_id: str | None = None) -> None:
        super().__init__(
            "Solución no disponible. El Solver debe procesar el problema primero.",
            details={"problem_id": problem_id} if problem_id else {},
            recoverable=True,
        )


# =============================================================================
# Errores de A2A Protocol
# =============================================================================

class A2AError(AulaError):
    """Error en el protocolo A2A."""
    pass


class A2AConnectionError(A2AError):
    """Error de conexión A2A."""
    
    def __init__(self, target_agent: str, endpoint: str, cause: str) -> None:
        super().__init__(
            f"No se pudo conectar con {target_agent} en {endpoint}",
            details={"target_agent": target_agent, "endpoint": endpoint, "cause": cause},
            recoverable=True,
        )


class A2ATimeoutError(A2AError):
    """Timeout en comunicación A2A."""
    
    def __init__(self, target_agent: str, timeout_seconds: float) -> None:
        super().__init__(
            f"Timeout de {timeout_seconds}s esperando respuesta de {target_agent}",
            details={"target_agent": target_agent, "timeout_seconds": timeout_seconds},
            recoverable=True,
        )


class A2AProtocolError(A2AError):
    """Error de protocolo A2A (mensaje malformado, etc.)."""
    
    def __init__(self, message: str, raw_message: dict[str, Any] | None = None) -> None:
        details = {}
        if raw_message:
            details["raw_message"] = str(raw_message)[:500]
        super().__init__(message, details=details, recoverable=False)


# =============================================================================
# Errores de Embedding
# =============================================================================

class EmbeddingError(AulaError):
    """Error en generación de embeddings."""
    pass


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Las dimensiones de embeddings no coinciden."""
    
    def __init__(self, expected: int, received: int) -> None:
        super().__init__(
            f"Dimensiones de embedding no coinciden. Esperado: {expected}, Recibido: {received}",
            details={"expected": expected, "received": received},
            recoverable=False,
        )


# =============================================================================
# Errores de Parsing
# =============================================================================

class ParsingError(AulaError):
    """Error al parsear respuestas del modelo."""
    pass


class JSONParsingError(ParsingError):
    """Error al parsear JSON de la respuesta."""
    
    def __init__(self, response_preview: str, parse_error: str) -> None:
        super().__init__(
            f"Error parseando JSON: {parse_error}",
            details={
                "response_preview": response_preview[:200],
                "parse_error": parse_error,
            },
            recoverable=True,
        )


class SolutionParsingError(ParsingError):
    """Error al parsear la estructura de solución."""
    
    def __init__(self, missing_fields: list[str] | None = None, cause: str | None = None) -> None:
        details = {}
        if missing_fields:
            details["missing_fields"] = missing_fields
        if cause:
            details["cause"] = cause
        super().__init__(
            "Error parseando estructura de solución",
            details=details,
            recoverable=True,
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base
    "AulaError",
    # Model errors
    "ModelError",
    "ModelNotFoundError",
    "ModelConnectionError",
    "ModelGenerationError",
    "ModelTimeoutError",
    "ModelLoadError",
    "InsufficientVRAMError",
    # Configuration errors
    "ConfigurationError",
    "InvalidModelIdError",
    "BackendNotSupportedError",
    # Guardrail errors
    "GuardrailError",
    "ManipulationDetectedError",
    "SolutionLeakDetectedError",
    "PedagogicalValidationError",
    # Agent errors
    "AgentError",
    "SolverError",
    "TutorError",
    "SolutionNotAvailableError",
    # A2A errors
    "A2AError",
    "A2AConnectionError",
    "A2ATimeoutError",
    "A2AProtocolError",
    # Embedding errors
    "EmbeddingError",
    "EmbeddingDimensionMismatchError",
    # Parsing errors
    "ParsingError",
    "JSONParsingError",
    "SolutionParsingError",
]
