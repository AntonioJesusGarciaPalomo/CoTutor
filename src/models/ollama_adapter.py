"""
Adaptador para modelos de Ollama.

Este módulo implementa la interfaz BaseModelAdapter para el backend Ollama,
permitiendo usar cualquier modelo disponible en Ollama de forma transparente.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.exceptions import (
    ModelConnectionError,
    ModelGenerationError,
    ModelNotFoundError,
    ModelTimeoutError,
)
from src.core.types import EmbeddingResponse, Message, ModelResponse
from src.models.base import BaseEmbeddingAdapter, BaseModelAdapter


class OllamaAdapter(BaseModelAdapter):
    """
    Adaptador para modelos de Ollama.
    
    Ollama proporciona una API REST local para ejecutar modelos LLM.
    Este adaptador soporta tanto generación como embeddings.
    
    Attributes:
        base_url: URL base del servidor Ollama.
        timeout: Timeout para las peticiones HTTP.
    
    Example:
        ```python
        adapter = OllamaAdapter(
            model_name="llama3.1:8b",
            base_url="http://localhost:11434",
        )
        
        response = await adapter.generate([
            {"role": "user", "content": "Hola, ¿cómo estás?"}
        ])
        print(response.content)
        ```
    """
    
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        """
        Inicializa el adaptador de Ollama.
        
        Args:
            model_name: Nombre del modelo en Ollama (ej: "llama3.1:8b").
            base_url: URL base del servidor Ollama.
            timeout: Timeout para peticiones en segundos.
            **kwargs: Argumentos adicionales.
        """
        super().__init__(
            model_name=model_name,
            backend_name="ollama",
            **kwargs,
        )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.supports_streaming = True
        self.supports_embeddings = True
        
        # Cliente HTTP asíncrono
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Obtiene o crea el cliente HTTP."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client
    
    async def _close_client(self) -> None:
        """Cierra el cliente HTTP."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def generate(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Genera una respuesta usando la API de chat de Ollama.
        
        Args:
            messages: Lista de mensajes de la conversación.
            temperature: Temperatura de sampling (0.0 - 2.0).
            max_tokens: Máximo de tokens a generar.
            stop: Secuencias de parada opcionales.
            system_prompt: Prompt de sistema opcional.
            **kwargs: Argumentos adicionales para Ollama.
            
        Returns:
            ModelResponse con el texto generado.
            
        Raises:
            ModelConnectionError: Si no se puede conectar a Ollama.
            ModelNotFoundError: Si el modelo no existe.
            ModelGenerationError: Si hay un error durante la generación.
            ModelTimeoutError: Si se excede el timeout.
        """
        client = await self._get_client()
        normalized_messages = self._normalize_messages(messages)
        
        # Añadir system prompt si se proporciona
        if system_prompt:
            normalized_messages = [
                {"role": "system", "content": system_prompt},
                *normalized_messages,
            ]
        
        # Construir payload
        payload = {
            "model": self.model_name,
            "messages": normalized_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        if stop:
            payload["options"]["stop"] = stop
        
        # Añadir opciones adicionales
        for key, value in kwargs.items():
            if key not in payload:
                payload["options"][key] = value
        
        start_time = time.perf_counter()
        
        try:
            response = await client.post("/api/chat", json=payload)
            
            if response.status_code == 404:
                raise ModelNotFoundError(self.model_name, "ollama")
            
            response.raise_for_status()
            data = response.json()
            
        except httpx.ConnectError as e:
            raise ModelConnectionError(
                "ollama",
                self.base_url,
                str(e),
            ) from e
        except httpx.TimeoutException as e:
            raise ModelTimeoutError(self.model_name, self.timeout) from e
        except httpx.HTTPStatusError as e:
            raise ModelGenerationError(
                self.model_name,
                f"HTTP {e.response.status_code}: {e.response.text}",
            ) from e
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Extraer contenido de la respuesta
        content = data.get("message", {}).get("content", "")
        
        # Métricas de tokens (Ollama las proporciona)
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        
        return self._create_response(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            generation_time_ms=elapsed_ms,
            finish_reason=data.get("done_reason", "stop"),
            raw_response=data,
        )
    
    async def generate_stream(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Genera una respuesta con streaming.
        
        Args:
            messages: Lista de mensajes de la conversación.
            temperature: Temperatura de sampling.
            max_tokens: Máximo de tokens a generar.
            stop: Secuencias de parada opcionales.
            system_prompt: Prompt de sistema opcional.
            **kwargs: Argumentos adicionales.
            
        Yields:
            Fragmentos de texto a medida que se generan.
        """
        client = await self._get_client()
        normalized_messages = self._normalize_messages(messages)
        
        if system_prompt:
            normalized_messages = [
                {"role": "system", "content": system_prompt},
                *normalized_messages,
            ]
        
        payload = {
            "model": self.model_name,
            "messages": normalized_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        if stop:
            payload["options"]["stop"] = stop
        
        for key, value in kwargs.items():
            if key not in payload:
                payload["options"][key] = value
        
        try:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code == 404:
                    raise ModelNotFoundError(self.model_name, "ollama")
                
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        import json
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                            
                            # Verificar si terminó
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.ConnectError as e:
            raise ModelConnectionError("ollama", self.base_url, str(e)) from e
        except httpx.TimeoutException as e:
            raise ModelTimeoutError(self.model_name, self.timeout) from e
    
    async def embed(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """
        Genera embeddings usando Ollama.
        
        Args:
            texts: Lista de textos a embeber.
            **kwargs: Argumentos adicionales.
            
        Returns:
            EmbeddingResponse con los vectores.
        """
        client = await self._get_client()
        embeddings = []
        total_tokens = 0
        
        start_time = time.perf_counter()
        
        for text in texts:
            payload = {
                "model": self.model_name,
                "input": text,
            }
            
            try:
                response = await client.post("/api/embed", json=payload)
                
                if response.status_code == 404:
                    raise ModelNotFoundError(self.model_name, "ollama")
                
                response.raise_for_status()
                data = response.json()
                
                # Ollama devuelve embeddings en diferentes formatos según versión
                embedding = data.get("embeddings", data.get("embedding", []))
                if isinstance(embedding, list) and len(embedding) > 0:
                    if isinstance(embedding[0], list):
                        embeddings.append(embedding[0])
                    else:
                        embeddings.append(embedding)
                
                # Contar tokens si está disponible
                if "prompt_eval_count" in data:
                    total_tokens += data["prompt_eval_count"]
                    
            except httpx.ConnectError as e:
                raise ModelConnectionError("ollama", self.base_url, str(e)) from e
            except httpx.HTTPStatusError as e:
                raise ModelGenerationError(
                    self.model_name,
                    f"Error en embeddings: {e.response.text}",
                ) from e
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Determinar dimensiones
        dimensions = len(embeddings[0]) if embeddings else 0
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=self.model_id,
            dimensions=dimensions,
            total_tokens=total_tokens if total_tokens > 0 else None,
            generation_time_ms=elapsed_ms,
        )
    
    async def health_check(self) -> bool:
        """
        Verifica si Ollama está disponible.
        
        Returns:
            True si Ollama está funcionando.
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def get_model_info(self) -> dict[str, Any]:
        """
        Obtiene información del modelo desde Ollama.
        
        Returns:
            Diccionario con información del modelo.
        """
        client = await self._get_client()
        
        try:
            response = await client.post("/api/show", json={"name": self.model_name})
            
            if response.status_code == 404:
                raise ModelNotFoundError(self.model_name, "ollama")
            
            response.raise_for_status()
            data = response.json()
            
            # Extraer información relevante
            modelinfo = data.get("modelinfo", {})
            
            return {
                "model": self.model_name,
                "backend": "ollama",
                "family": modelinfo.get("general.architecture"),
                "parameter_size": data.get("details", {}).get("parameter_size"),
                "quantization": data.get("details", {}).get("quantization_level"),
                "context_length": modelinfo.get(
                    "llama.context_length",
                    modelinfo.get("context_length", 4096)
                ),
                "license": data.get("license"),
            }
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(self.model_name, "ollama")
            raise
    
    async def list_models(self) -> list[dict[str, Any]]:
        """
        Lista todos los modelos disponibles en Ollama.
        
        Returns:
            Lista de diccionarios con información de cada modelo.
        """
        client = await self._get_client()
        
        try:
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            
            return [
                {
                    "name": model.get("name"),
                    "size": model.get("size"),
                    "modified_at": model.get("modified_at"),
                    "digest": model.get("digest"),
                }
                for model in data.get("models", [])
            ]
            
        except httpx.ConnectError as e:
            raise ModelConnectionError("ollama", self.base_url, str(e)) from e
    
    async def pull_model(self, model_name: str | None = None) -> None:
        """
        Descarga un modelo en Ollama.
        
        Args:
            model_name: Nombre del modelo a descargar. Si es None, usa self.model_name.
        """
        client = await self._get_client()
        name = model_name or self.model_name
        
        try:
            # Pull es una operación de larga duración
            async with client.stream(
                "POST",
                "/api/pull",
                json={"name": name},
                timeout=httpx.Timeout(None),  # Sin timeout
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    # Podríamos loggear el progreso aquí
                    pass
                    
        except httpx.HTTPStatusError as e:
            raise ModelGenerationError(
                name,
                f"Error descargando modelo: {e.response.text}",
            ) from e
    
    async def count_tokens(self, text: str) -> int:
        """
        Cuenta tokens usando Ollama (si está disponible).
        
        Ollama no tiene un endpoint dedicado para tokenización,
        así que usamos una aproximación o el conteo del prompt.
        """
        # Aproximación: ~4 caracteres por token
        return len(text) // 4
    
    async def unload(self) -> None:
        """Cierra la conexión con Ollama."""
        await self._close_client()
        self.is_loaded = False
    
    def __del__(self) -> None:
        """Limpieza al destruir el objeto."""
        if self._client is not None and not self._client.is_closed:
            # No podemos usar await en __del__, crear tarea
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._close_client())
            except RuntimeError:
                pass


class OllamaEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Adaptador dedicado para embeddings de Ollama.
    
    Útil cuando se quiere usar un modelo de embeddings diferente
    al modelo principal de generación.
    
    Example:
        ```python
        embedder = OllamaEmbeddingAdapter(
            model_name="nomic-embed-text",
        )
        
        response = await embedder.embed(["Texto de ejemplo"])
        print(f"Dimensiones: {response.dimensions}")
        ```
    """
    
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            backend_name="ollama",
            dimensions=dimensions,
            **kwargs,
        )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Obtiene o crea el cliente HTTP."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client
    
    async def embed(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """
        Genera embeddings para una lista de textos.
        
        Args:
            texts: Lista de textos.
            **kwargs: Argumentos adicionales.
            
        Returns:
            EmbeddingResponse con los vectores.
        """
        client = await self._get_client()
        embeddings = []
        total_tokens = 0
        
        start_time = time.perf_counter()
        
        for text in texts:
            payload = {
                "model": self.model_name,
                "input": text,
            }
            
            try:
                response = await client.post("/api/embed", json=payload)
                
                if response.status_code == 404:
                    raise ModelNotFoundError(self.model_name, "ollama")
                
                response.raise_for_status()
                data = response.json()
                
                embedding = data.get("embeddings", data.get("embedding", []))
                if isinstance(embedding, list) and len(embedding) > 0:
                    if isinstance(embedding[0], list):
                        embeddings.append(embedding[0])
                    else:
                        embeddings.append(embedding)
                
                if "prompt_eval_count" in data:
                    total_tokens += data["prompt_eval_count"]
                    
            except httpx.ConnectError as e:
                raise ModelConnectionError("ollama", self.base_url, str(e)) from e
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Actualizar dimensiones si no se conocían
        if embeddings and self.dimensions is None:
            self.dimensions = len(embeddings[0])
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=self.model_id,
            dimensions=self.dimensions or len(embeddings[0]) if embeddings else 0,
            total_tokens=total_tokens if total_tokens > 0 else None,
            generation_time_ms=elapsed_ms,
        )
    
    async def health_check(self) -> bool:
        """Verifica si el servicio está disponible."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False


__all__ = [
    "OllamaAdapter",
    "OllamaEmbeddingAdapter",
]
