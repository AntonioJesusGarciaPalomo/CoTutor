"""
Interfaz abstracta para modelos de lenguaje.

Este módulo define el contrato que todos los adaptadores de modelos
deben implementar, permitiendo intercambiar backends de forma transparente.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, AsyncIterator

from src.core.types import EmbeddingResponse, Message, MessageRole, ModelResponse


class BaseModelAdapter(ABC):
    """
    Clase base abstracta para adaptadores de modelos de lenguaje.
    
    Todos los backends (Ollama, OpenAI local, HuggingFace) deben heredar
    de esta clase e implementar sus métodos abstractos.
    
    Attributes:
        model_name: Nombre del modelo específico.
        backend_name: Nombre del backend (ollama, openai_local, huggingface).
        is_loaded: Indica si el modelo está cargado y listo.
        supports_streaming: Indica si el backend soporta streaming.
        supports_embeddings: Indica si el backend soporta generación de embeddings.
    """
    
    def __init__(
        self,
        model_name: str,
        backend_name: str,
        **kwargs: Any,
    ) -> None:
        """
        Inicializa el adaptador base.
        
        Args:
            model_name: Nombre del modelo a usar.
            backend_name: Identificador del backend.
            **kwargs: Argumentos adicionales específicos del backend.
        """
        self.model_name = model_name
        self.backend_name = backend_name
        self.is_loaded = False
        self.supports_streaming = True
        self.supports_embeddings = False
        self._config = kwargs
    
    @property
    def model_id(self) -> str:
        """Identificador completo del modelo (backend/model_name)."""
        return f"{self.backend_name}/{self.model_name}"
    
    # =========================================================================
    # Métodos abstractos que deben implementar los adaptadores
    # =========================================================================
    
    @abstractmethod
    async def generate(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Genera una respuesta completa (sin streaming).
        
        Args:
            messages: Lista de mensajes de la conversación.
            temperature: Temperatura de sampling (0.0 - 2.0).
            max_tokens: Máximo de tokens a generar.
            stop: Secuencias de parada opcionales.
            **kwargs: Argumentos adicionales del backend.
            
        Returns:
            ModelResponse con el texto generado y métricas.
            
        Raises:
            ModelGenerationError: Si hay un error durante la generación.
            ModelTimeoutError: Si se excede el timeout.
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Genera una respuesta con streaming.
        
        Args:
            messages: Lista de mensajes de la conversación.
            temperature: Temperatura de sampling.
            max_tokens: Máximo de tokens a generar.
            stop: Secuencias de parada opcionales.
            **kwargs: Argumentos adicionales del backend.
            
        Yields:
            Fragmentos de texto a medida que se generan.
            
        Raises:
            ModelGenerationError: Si hay un error durante la generación.
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Verifica si el backend está disponible y funcionando.
        
        Returns:
            True si el backend está operativo, False en caso contrario.
        """
        pass
    
    @abstractmethod
    async def get_model_info(self) -> dict[str, Any]:
        """
        Obtiene información sobre el modelo cargado.
        
        Returns:
            Diccionario con información del modelo (context_length, etc.)
        """
        pass
    
    # =========================================================================
    # Métodos opcionales (con implementación por defecto)
    # =========================================================================
    
    async def embed(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """
        Genera embeddings para una lista de textos.
        
        Args:
            texts: Lista de textos a embeber.
            **kwargs: Argumentos adicionales.
            
        Returns:
            EmbeddingResponse con los vectores de embedding.
            
        Raises:
            NotImplementedError: Si el backend no soporta embeddings.
        """
        raise NotImplementedError(
            f"El backend {self.backend_name} no soporta generación de embeddings. "
            f"Use un modelo de embeddings dedicado."
        )
    
    async def count_tokens(self, text: str) -> int:
        """
        Cuenta los tokens en un texto.
        
        Implementación por defecto usando aproximación.
        Los backends que soporten conteo preciso deben sobrescribir este método.
        
        Args:
            text: Texto a tokenizar.
            
        Returns:
            Número estimado de tokens.
        """
        # Aproximación: ~4 caracteres por token en inglés/español
        return len(text) // 4
    
    async def load(self) -> None:
        """
        Carga el modelo en memoria (para backends que lo requieran).
        
        Por defecto no hace nada. Los backends como HuggingFace
        deben sobrescribir este método.
        """
        self.is_loaded = True
    
    async def unload(self) -> None:
        """
        Descarga el modelo de memoria.
        
        Por defecto no hace nada. Los backends como HuggingFace
        deben sobrescribir este método.
        """
        self.is_loaded = False
    
    # =========================================================================
    # Métodos de utilidad
    # =========================================================================
    
    def _normalize_messages(
        self,
        messages: list[Message] | list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        Normaliza mensajes al formato de diccionario estándar.
        
        Args:
            messages: Mensajes en formato Message o dict.
            
        Returns:
            Lista de diccionarios con 'role' y 'content'.
        """
        normalized = []
        for msg in messages:
            if isinstance(msg, Message):
                normalized.append({
                    "role": msg.role.value if isinstance(msg.role, MessageRole) else msg.role,
                    "content": msg.content,
                })
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                if isinstance(role, MessageRole):
                    role = role.value
                normalized.append({
                    "role": role,
                    "content": msg.get("content", ""),
                })
            else:
                raise ValueError(f"Formato de mensaje no soportado: {type(msg)}")
        return normalized
    
    def _create_response(
        self,
        content: str,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        generation_time_ms: float | None = None,
        finish_reason: str | None = None,
        raw_response: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """
        Crea un objeto ModelResponse estandarizado.
        
        Args:
            content: Texto generado.
            prompt_tokens: Tokens del prompt.
            completion_tokens: Tokens generados.
            generation_time_ms: Tiempo de generación en ms.
            finish_reason: Razón de finalización.
            raw_response: Respuesta raw del backend.
            
        Returns:
            ModelResponse configurado.
        """
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        
        tokens_per_second = None
        if completion_tokens and generation_time_ms and generation_time_ms > 0:
            tokens_per_second = completion_tokens / (generation_time_ms / 1000)
        
        return ModelResponse(
            content=content,
            model=self.model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            generation_time_ms=generation_time_ms,
            tokens_per_second=tokens_per_second,
            finish_reason=finish_reason,
            raw_response=raw_response,
        )
    
    @asynccontextmanager
    async def _measure_time(self) -> AsyncGenerator[dict[str, float], None]:
        """
        Context manager para medir tiempo de ejecución.
        
        Yields:
            Diccionario donde se guardará 'elapsed_ms'.
        """
        timing: dict[str, float] = {}
        start = time.perf_counter()
        try:
            yield timing
        finally:
            elapsed = time.perf_counter() - start
            timing["elapsed_ms"] = elapsed * 1000
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_id}, loaded={self.is_loaded})"


class BaseEmbeddingAdapter(ABC):
    """
    Clase base para adaptadores de modelos de embedding.
    
    Separada de BaseModelAdapter porque algunos backends tienen
    modelos de embedding dedicados.
    """
    
    def __init__(
        self,
        model_name: str,
        backend_name: str,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Inicializa el adaptador de embeddings.
        
        Args:
            model_name: Nombre del modelo de embeddings.
            backend_name: Identificador del backend.
            dimensions: Dimensiones del embedding (si se conoce).
            **kwargs: Argumentos adicionales.
        """
        self.model_name = model_name
        self.backend_name = backend_name
        self.dimensions = dimensions
        self._config = kwargs
    
    @property
    def model_id(self) -> str:
        """Identificador completo del modelo."""
        return f"{self.backend_name}/{self.model_name}"
    
    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """
        Genera embeddings para una lista de textos.
        
        Args:
            texts: Lista de textos a embeber.
            **kwargs: Argumentos adicionales.
            
        Returns:
            EmbeddingResponse con los vectores.
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Verifica si el servicio de embeddings está disponible."""
        pass
    
    async def embed_single(self, text: str, **kwargs: Any) -> list[float]:
        """
        Genera embedding para un solo texto.
        
        Convenience method que llama a embed() con un solo texto.
        
        Args:
            text: Texto a embeber.
            **kwargs: Argumentos adicionales.
            
        Returns:
            Vector de embedding.
        """
        response = await self.embed([text], **kwargs)
        return response.embeddings[0]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_id}, dims={self.dimensions})"


# =============================================================================
# Type hints para Factory
# =============================================================================

# Tipo para funciones factory que crean adaptadores
ModelAdapterFactory = type[BaseModelAdapter]
EmbeddingAdapterFactory = type[BaseEmbeddingAdapter]


__all__ = [
    "BaseModelAdapter",
    "BaseEmbeddingAdapter",
    "ModelAdapterFactory",
    "EmbeddingAdapterFactory",
]
