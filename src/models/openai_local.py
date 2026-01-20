"""
Adaptador para servidores compatibles con la API de OpenAI.

Este módulo permite usar servidores locales que implementan la API de OpenAI,
como vLLM, llama.cpp server, LM Studio, LocalAI, etc.
"""

from __future__ import annotations

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


class OpenAILocalAdapter(BaseModelAdapter):
    """
    Adaptador para servidores que implementan la API de OpenAI.
    
    Soporta:
    - vLLM (http://localhost:8000/v1)
    - llama.cpp server (http://localhost:8080/v1)
    - LM Studio (http://localhost:1234/v1)
    - LocalAI (http://localhost:8080/v1)
    - Text Generation WebUI con extensión openai
    - Cualquier otro servidor compatible con OpenAI API
    
    Attributes:
        base_url: URL base del servidor (ej: "http://localhost:8000/v1").
        api_key: API key (generalmente no requerida para servidores locales).
        timeout: Timeout para las peticiones HTTP.
    
    Example:
        ```python
        # Para vLLM
        adapter = OpenAILocalAdapter(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            base_url="http://localhost:8000/v1",
        )
        
        # Para llama.cpp
        adapter = OpenAILocalAdapter(
            model_name="loaded-model",  # llama.cpp usa un modelo fijo
            base_url="http://localhost:8080/v1",
        )
        
        response = await adapter.generate([
            {"role": "user", "content": "Hola!"}
        ])
        ```
    """
    
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        timeout: float = 120.0,
        server_type: str = "generic",
        **kwargs: Any,
    ) -> None:
        """
        Inicializa el adaptador OpenAI Local.
        
        Args:
            model_name: Nombre del modelo (varía según el servidor).
            base_url: URL base del servidor con /v1.
            api_key: API key (usar "not-needed" si no se requiere).
            timeout: Timeout en segundos.
            server_type: Tipo de servidor (vllm, llamacpp, lmstudio, generic).
            **kwargs: Argumentos adicionales.
        """
        super().__init__(
            model_name=model_name,
            backend_name="openai_local",
            **kwargs,
        )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.server_type = server_type
        self.supports_streaming = True
        self.supports_embeddings = server_type in ("vllm", "generic")
        
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Obtiene o crea el cliente HTTP con headers de autenticación."""
        if self._client is None or self._client.is_closed:
            headers = {
                "Content-Type": "application/json",
            }
            if self.api_key and self.api_key != "not-needed":
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
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
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Genera una respuesta usando la API de chat completions.
        
        Args:
            messages: Lista de mensajes de la conversación.
            temperature: Temperatura de sampling (0.0 - 2.0).
            max_tokens: Máximo de tokens a generar.
            stop: Secuencias de parada opcionales.
            top_p: Nucleus sampling.
            frequency_penalty: Penalización por frecuencia.
            presence_penalty: Penalización por presencia.
            **kwargs: Argumentos adicionales.
            
        Returns:
            ModelResponse con el texto generado.
        """
        client = await self._get_client()
        normalized_messages = self._normalize_messages(messages)
        
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": normalized_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": False,
        }
        
        if stop:
            payload["stop"] = stop
        
        # Añadir parámetros adicionales específicos del servidor
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value
        
        start_time = time.perf_counter()
        
        try:
            response = await client.post("/chat/completions", json=payload)
            
            # Manejar errores de modelo no encontrado
            if response.status_code == 404:
                raise ModelNotFoundError(self.model_name, "openai_local")
            
            if response.status_code == 400:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", response.text)
                raise ModelGenerationError(self.model_name, error_msg)
            
            response.raise_for_status()
            data = response.json()
            
        except httpx.ConnectError as e:
            raise ModelConnectionError(
                "openai_local",
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
        
        # Extraer respuesta del formato OpenAI
        choices = data.get("choices", [])
        if not choices:
            raise ModelGenerationError(
                self.model_name,
                "No se recibieron choices en la respuesta",
            )
        
        content = choices[0].get("message", {}).get("content", "")
        finish_reason = choices[0].get("finish_reason")
        
        # Métricas de uso
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        
        return self._create_response(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            generation_time_ms=elapsed_ms,
            finish_reason=finish_reason,
            raw_response=data,
        )
    
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
        Genera una respuesta con streaming usando SSE.
        
        Args:
            messages: Lista de mensajes.
            temperature: Temperatura de sampling.
            max_tokens: Máximo de tokens.
            stop: Secuencias de parada.
            **kwargs: Argumentos adicionales.
            
        Yields:
            Fragmentos de texto.
        """
        client = await self._get_client()
        normalized_messages = self._normalize_messages(messages)
        
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": normalized_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        
        if stop:
            payload["stop"] = stop
        
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value
        
        try:
            async with client.stream(
                "POST",
                "/chat/completions",
                json=payload,
            ) as response:
                if response.status_code == 404:
                    raise ModelNotFoundError(self.model_name, "openai_local")
                
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Quitar "data: "
                        
                        if data_str.strip() == "[DONE]":
                            break
                        
                        import json
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.ConnectError as e:
            raise ModelConnectionError("openai_local", self.base_url, str(e)) from e
        except httpx.TimeoutException as e:
            raise ModelTimeoutError(self.model_name, self.timeout) from e
    
    async def embed(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """
        Genera embeddings usando el endpoint /embeddings.
        
        Solo disponible en servidores que lo soporten (vLLM, LocalAI).
        
        Args:
            texts: Lista de textos.
            **kwargs: Argumentos adicionales.
            
        Returns:
            EmbeddingResponse con los vectores.
        """
        client = await self._get_client()
        
        payload = {
            "model": self.model_name,
            "input": texts,
        }
        
        start_time = time.perf_counter()
        
        try:
            response = await client.post("/embeddings", json=payload)
            
            if response.status_code == 404:
                raise ModelNotFoundError(self.model_name, "openai_local")
            
            response.raise_for_status()
            data = response.json()
            
        except httpx.ConnectError as e:
            raise ModelConnectionError("openai_local", self.base_url, str(e)) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise NotImplementedError(
                    f"El servidor {self.server_type} en {self.base_url} "
                    f"no soporta el endpoint /embeddings"
                ) from e
            raise ModelGenerationError(
                self.model_name,
                f"Error en embeddings: {e.response.text}",
            ) from e
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Extraer embeddings del formato OpenAI
        embeddings = [item["embedding"] for item in data.get("data", [])]
        dimensions = len(embeddings[0]) if embeddings else 0
        
        usage = data.get("usage", {})
        total_tokens = usage.get("total_tokens")
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=self.model_id,
            dimensions=dimensions,
            total_tokens=total_tokens,
            generation_time_ms=elapsed_ms,
        )
    
    async def health_check(self) -> bool:
        """
        Verifica si el servidor está disponible.
        
        Intenta listar modelos o hacer una petición simple.
        """
        try:
            client = await self._get_client()
            
            # Intentar listar modelos (endpoint estándar)
            response = await client.get("/models")
            
            if response.status_code == 200:
                return True
            
            # Algunos servidores no implementan /models
            # Intentar un health check alternativo
            if response.status_code == 404:
                # Intentar una generación mínima
                test_payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                }
                response = await client.post("/chat/completions", json=test_payload)
                return response.status_code == 200
            
            return False
            
        except Exception:
            return False
    
    async def get_model_info(self) -> dict[str, Any]:
        """
        Obtiene información del modelo.
        
        La información disponible depende del servidor.
        """
        client = await self._get_client()
        
        try:
            response = await client.get("/models")
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                
                # Buscar nuestro modelo en la lista
                for model in models:
                    if model.get("id") == self.model_name:
                        return {
                            "model": self.model_name,
                            "backend": "openai_local",
                            "server_type": self.server_type,
                            "owned_by": model.get("owned_by"),
                            "created": model.get("created"),
                        }
                
                # Si no encontramos el modelo específico, devolver info genérica
                return {
                    "model": self.model_name,
                    "backend": "openai_local",
                    "server_type": self.server_type,
                    "available_models": [m.get("id") for m in models],
                }
            
            return {
                "model": self.model_name,
                "backend": "openai_local",
                "server_type": self.server_type,
            }
            
        except httpx.HTTPStatusError:
            return {
                "model": self.model_name,
                "backend": "openai_local",
                "server_type": self.server_type,
            }
    
    async def list_models(self) -> list[dict[str, Any]]:
        """
        Lista los modelos disponibles en el servidor.
        
        Returns:
            Lista de modelos con su información.
        """
        client = await self._get_client()
        
        try:
            response = await client.get("/models")
            response.raise_for_status()
            data = response.json()
            
            return [
                {
                    "id": model.get("id"),
                    "owned_by": model.get("owned_by"),
                    "created": model.get("created"),
                }
                for model in data.get("data", [])
            ]
            
        except Exception:
            # Muchos servidores no implementan este endpoint
            return [{"id": self.model_name, "note": "endpoint /models not available"}]
    
    async def unload(self) -> None:
        """Cierra la conexión."""
        await self._close_client()
        self.is_loaded = False


class OpenAILocalEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Adaptador dedicado para embeddings en servidores OpenAI-compatible.
    
    Útil cuando se usa un modelo de embeddings diferente al de generación.
    """
    
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        timeout: float = 60.0,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            backend_name="openai_local",
            dimensions=dimensions,
            **kwargs,
        )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Obtiene o crea el cliente HTTP."""
        if self._client is None or self._client.is_closed:
            headers = {"Content-Type": "application/json"}
            if self.api_key and self.api_key != "not-needed":
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client
    
    async def embed(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Genera embeddings."""
        client = await self._get_client()
        
        payload = {
            "model": self.model_name,
            "input": texts,
        }
        
        start_time = time.perf_counter()
        
        try:
            response = await client.post("/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
            
        except httpx.ConnectError as e:
            raise ModelConnectionError("openai_local", self.base_url, str(e)) from e
        except httpx.HTTPStatusError as e:
            raise ModelGenerationError(
                self.model_name,
                f"Error en embeddings: {e.response.text}",
            ) from e
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        embeddings = [item["embedding"] for item in data.get("data", [])]
        
        if embeddings and self.dimensions is None:
            self.dimensions = len(embeddings[0])
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=self.model_id,
            dimensions=self.dimensions or len(embeddings[0]) if embeddings else 0,
            total_tokens=data.get("usage", {}).get("total_tokens"),
            generation_time_ms=elapsed_ms,
        )
    
    async def health_check(self) -> bool:
        """Verifica disponibilidad."""
        try:
            client = await self._get_client()
            response = await client.get("/models")
            return response.status_code == 200
        except Exception:
            return False


__all__ = [
    "OpenAILocalAdapter",
    "OpenAILocalEmbeddingAdapter",
]
