"""
Factory para crear adaptadores de modelos.

Este módulo proporciona el patrón Factory para instanciar adaptadores
de forma transparente, permitiendo cambiar de backend con solo
modificar el identificador del modelo.
"""

from __future__ import annotations

import asyncio
from typing import Any, TypeVar

from config.settings import ModelBackend, get_settings, parse_model_id
from src.core.exceptions import (
    BackendNotSupportedError,
    ConfigurationError,
    ModelConnectionError,
)
from src.models.base import BaseEmbeddingAdapter, BaseModelAdapter
from src.models.huggingface_local import (
    HuggingFaceEmbeddingAdapter,
    HuggingFaceLocalAdapter,
)
from src.models.ollama_adapter import OllamaAdapter, OllamaEmbeddingAdapter
from src.models.openai_local import OpenAILocalAdapter, OpenAILocalEmbeddingAdapter


# Type variable para adaptadores
T = TypeVar("T", bound=BaseModelAdapter)
E = TypeVar("E", bound=BaseEmbeddingAdapter)


# Registro de backends soportados
SUPPORTED_BACKENDS = ["ollama", "openai_local", "huggingface"]


class ModelFactory:
    """
    Factory para crear adaptadores de modelos de lenguaje.
    
    Esta clase proporciona métodos estáticos para crear adaptadores
    de forma transparente, basándose en el identificador del modelo.
    
    El formato del identificador es: "backend/model_name"
    Ejemplos:
    - "ollama/llama3.1:8b"
    - "openai_local/meta-llama/Llama-3.1-70B-Instruct"
    - "huggingface/Qwen/Qwen2.5-14B-Instruct"
    
    Example:
        ```python
        # Crear un adaptador de Ollama
        adapter = ModelFactory.create("ollama/qwen2.5:14b")
        
        # Crear un adaptador de vLLM
        adapter = ModelFactory.create(
            "openai_local/meta-llama/Llama-3.1-8B-Instruct",
            base_url="http://localhost:8000/v1",
        )
        
        # Crear un adaptador de HuggingFace
        adapter = ModelFactory.create(
            "huggingface/Qwen/Qwen2.5-14B-Instruct",
            load_in_4bit=True,
        )
        ```
    """
    
    # Cache de adaptadores (singleton por model_id)
    _cache: dict[str, BaseModelAdapter] = {}
    _embedding_cache: dict[str, BaseEmbeddingAdapter] = {}
    
    @staticmethod
    def create(
        model_id: str,
        *,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> BaseModelAdapter:
        """
        Crea un adaptador de modelo basado en el identificador.
        
        Args:
            model_id: Identificador del modelo ("backend/model_name").
            use_cache: Si usar caché de adaptadores (singleton).
            **kwargs: Argumentos adicionales para el adaptador.
            
        Returns:
            Instancia del adaptador configurado.
            
        Raises:
            InvalidModelIdError: Si el formato del model_id es inválido.
            BackendNotSupportedError: Si el backend no está soportado.
            ConfigurationError: Si hay un error de configuración.
        """
        # Verificar caché
        cache_key = f"{model_id}:{hash(frozenset(kwargs.items()))}"
        if use_cache and cache_key in ModelFactory._cache:
            return ModelFactory._cache[cache_key]
        
        # Parsear model_id
        backend, model_name = parse_model_id(model_id)
        
        # Verificar backend
        if backend not in SUPPORTED_BACKENDS:
            raise BackendNotSupportedError(backend, SUPPORTED_BACKENDS)
        
        # Obtener configuración
        settings = get_settings()
        
        # Crear adaptador según backend
        adapter: BaseModelAdapter
        
        if backend == "ollama":
            adapter = ModelFactory._create_ollama(model_name, settings, **kwargs)
        elif backend == "openai_local":
            adapter = ModelFactory._create_openai_local(model_name, settings, **kwargs)
        elif backend == "huggingface":
            adapter = ModelFactory._create_huggingface(model_name, settings, **kwargs)
        else:
            raise BackendNotSupportedError(backend, SUPPORTED_BACKENDS)
        
        # Guardar en caché
        if use_cache:
            ModelFactory._cache[cache_key] = adapter
        
        return adapter
    
    @staticmethod
    def _create_ollama(
        model_name: str,
        settings: Any,
        **kwargs: Any,
    ) -> OllamaAdapter:
        """Crea un adaptador de Ollama."""
        # Obtener configuración base
        base_url = kwargs.pop("base_url", settings.ollama.base_url)
        timeout = kwargs.pop("timeout", settings.ollama.timeout)
        
        return OllamaAdapter(
            model_name=model_name,
            base_url=base_url,
            timeout=timeout,
            **kwargs,
        )
    
    @staticmethod
    def _create_openai_local(
        model_name: str,
        settings: Any,
        **kwargs: Any,
    ) -> OpenAILocalAdapter:
        """Crea un adaptador OpenAI Local."""
        # Determinar servidor y configuración
        base_url = kwargs.pop("base_url", settings.openai_local.base_url)
        api_key = kwargs.pop("api_key", settings.openai_local.api_key)
        timeout = kwargs.pop("timeout", settings.openai_local.timeout)
        server_type = kwargs.pop("server_type", "generic")
        
        return OpenAILocalAdapter(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            server_type=server_type,
            **kwargs,
        )
    
    @staticmethod
    def _create_huggingface(
        model_name: str,
        settings: Any,
        **kwargs: Any,
    ) -> HuggingFaceLocalAdapter:
        """Crea un adaptador de HuggingFace."""
        # Obtener configuración
        hf_config = settings.huggingface
        
        device_map = kwargs.pop("device_map", hf_config.device_map)
        torch_dtype = kwargs.pop("torch_dtype", hf_config.torch_dtype)
        load_in_4bit = kwargs.pop("load_in_4bit", hf_config.load_in_4bit)
        load_in_8bit = kwargs.pop("load_in_8bit", hf_config.load_in_8bit)
        cache_dir = kwargs.pop("cache_dir", hf_config.cache_dir)
        
        return HuggingFaceLocalAdapter(
            model_name=model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            cache_dir=cache_dir,
            **kwargs,
        )
    
    @staticmethod
    def create_embedding(
        model_id: str,
        *,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> BaseEmbeddingAdapter:
        """
        Crea un adaptador de embeddings.
        
        Args:
            model_id: Identificador del modelo de embeddings.
            use_cache: Si usar caché.
            **kwargs: Argumentos adicionales.
            
        Returns:
            Instancia del adaptador de embeddings.
        """
        cache_key = f"emb:{model_id}:{hash(frozenset(kwargs.items()))}"
        if use_cache and cache_key in ModelFactory._embedding_cache:
            return ModelFactory._embedding_cache[cache_key]
        
        backend, model_name = parse_model_id(model_id)
        settings = get_settings()
        
        adapter: BaseEmbeddingAdapter
        
        if backend == "ollama":
            adapter = OllamaEmbeddingAdapter(
                model_name=model_name,
                base_url=kwargs.pop("base_url", settings.ollama.base_url),
                **kwargs,
            )
        elif backend == "openai_local":
            adapter = OpenAILocalEmbeddingAdapter(
                model_name=model_name,
                base_url=kwargs.pop("base_url", settings.openai_local.base_url),
                **kwargs,
            )
        elif backend == "huggingface":
            adapter = HuggingFaceEmbeddingAdapter(
                model_name=model_name,
                cache_dir=kwargs.pop("cache_dir", settings.huggingface.cache_dir),
                **kwargs,
            )
        else:
            raise BackendNotSupportedError(backend, SUPPORTED_BACKENDS)
        
        if use_cache:
            ModelFactory._embedding_cache[cache_key] = adapter
        
        return adapter
    
    @staticmethod
    async def create_and_verify(
        model_id: str,
        **kwargs: Any,
    ) -> BaseModelAdapter:
        """
        Crea un adaptador y verifica que el backend esté disponible.
        
        Esta función es útil para detectar errores de configuración
        tempranamente.
        
        Args:
            model_id: Identificador del modelo.
            **kwargs: Argumentos adicionales.
            
        Returns:
            Adaptador verificado.
            
        Raises:
            ModelConnectionError: Si el backend no está disponible.
        """
        adapter = ModelFactory.create(model_id, **kwargs)
        
        is_healthy = await adapter.health_check()
        if not is_healthy:
            backend, _ = parse_model_id(model_id)
            
            # Obtener URL según backend
            settings = get_settings()
            if backend == "ollama":
                url = settings.ollama.base_url
            elif backend == "openai_local":
                url = settings.openai_local.base_url
            else:
                url = "local"
            
            raise ModelConnectionError(
                backend,
                url,
                f"Health check falló para {model_id}",
            )
        
        return adapter
    
    @staticmethod
    def clear_cache() -> None:
        """Limpia la caché de adaptadores."""
        ModelFactory._cache.clear()
        ModelFactory._embedding_cache.clear()
    
    @staticmethod
    async def cleanup_all() -> None:
        """Cierra todos los adaptadores en caché."""
        for adapter in ModelFactory._cache.values():
            await adapter.unload()
        ModelFactory._cache.clear()
        
        ModelFactory._embedding_cache.clear()
    
    @staticmethod
    def list_cached() -> list[str]:
        """Lista los adaptadores en caché."""
        return list(ModelFactory._cache.keys())


class ModelManager:
    """
    Gestor de alto nivel para modelos.
    
    Proporciona una interfaz simplificada para gestionar múltiples
    modelos, con soporte para precargar, caché y limpieza automática.
    
    Example:
        ```python
        manager = ModelManager()
        
        # Precargar modelos
        await manager.preload([
            "ollama/qwen2.5:14b",
            "ollama/llama3.1:8b",
        ])
        
        # Obtener un modelo (desde caché si existe)
        solver_model = await manager.get("ollama/qwen2.5:14b")
        tutor_model = await manager.get("ollama/llama3.1:8b")
        
        # Limpiar al terminar
        await manager.cleanup()
        ```
    """
    
    def __init__(self) -> None:
        """Inicializa el gestor de modelos."""
        self._models: dict[str, BaseModelAdapter] = {}
        self._embeddings: dict[str, BaseEmbeddingAdapter] = {}
        self._lock = asyncio.Lock()
    
    async def get(
        self,
        model_id: str,
        **kwargs: Any,
    ) -> BaseModelAdapter:
        """
        Obtiene un adaptador de modelo.
        
        Si el modelo no está en caché, lo crea y opcionalmente
        verifica su disponibilidad.
        
        Args:
            model_id: Identificador del modelo.
            **kwargs: Argumentos adicionales.
            
        Returns:
            Adaptador del modelo.
        """
        async with self._lock:
            if model_id not in self._models:
                adapter = await ModelFactory.create_and_verify(model_id, **kwargs)
                self._models[model_id] = adapter
            return self._models[model_id]
    
    async def get_embedding(
        self,
        model_id: str,
        **kwargs: Any,
    ) -> BaseEmbeddingAdapter:
        """
        Obtiene un adaptador de embeddings.
        
        Args:
            model_id: Identificador del modelo.
            **kwargs: Argumentos adicionales.
            
        Returns:
            Adaptador de embeddings.
        """
        async with self._lock:
            if model_id not in self._embeddings:
                adapter = ModelFactory.create_embedding(model_id, **kwargs)
                self._embeddings[model_id] = adapter
            return self._embeddings[model_id]
    
    async def preload(
        self,
        model_ids: list[str],
        verify: bool = True,
    ) -> dict[str, bool]:
        """
        Precarga múltiples modelos.
        
        Args:
            model_ids: Lista de identificadores de modelos.
            verify: Si verificar disponibilidad de cada modelo.
            
        Returns:
            Diccionario con el resultado de carga de cada modelo.
        """
        results = {}
        
        for model_id in model_ids:
            try:
                if verify:
                    await self.get(model_id)
                else:
                    adapter = ModelFactory.create(model_id)
                    self._models[model_id] = adapter
                results[model_id] = True
            except Exception as e:
                results[model_id] = False
                # Log error pero continuar con otros modelos
        
        return results
    
    async def health_check_all(self) -> dict[str, bool]:
        """
        Verifica el estado de todos los modelos cargados.
        
        Returns:
            Diccionario con el estado de cada modelo.
        """
        results = {}
        
        for model_id, adapter in self._models.items():
            results[model_id] = await adapter.health_check()
        
        for model_id, adapter in self._embeddings.items():
            results[f"embedding:{model_id}"] = await adapter.health_check()
        
        return results
    
    async def cleanup(self) -> None:
        """Limpia todos los modelos cargados."""
        async with self._lock:
            for adapter in self._models.values():
                await adapter.unload()
            self._models.clear()
            self._embeddings.clear()
    
    def list_loaded(self) -> list[str]:
        """Lista los modelos actualmente cargados."""
        return list(self._models.keys())
    
    def __contains__(self, model_id: str) -> bool:
        """Verifica si un modelo está cargado."""
        return model_id in self._models


# Singleton del gestor de modelos
_model_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """
    Obtiene el singleton del gestor de modelos.
    
    Returns:
        Instancia global del ModelManager.
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# Funciones de conveniencia
async def get_model(model_id: str, **kwargs: Any) -> BaseModelAdapter:
    """
    Obtiene un modelo usando el gestor global.
    
    Shortcut para get_model_manager().get(model_id).
    """
    return await get_model_manager().get(model_id, **kwargs)


async def get_embedding_model(model_id: str, **kwargs: Any) -> BaseEmbeddingAdapter:
    """
    Obtiene un modelo de embeddings usando el gestor global.
    """
    return await get_model_manager().get_embedding(model_id, **kwargs)


__all__ = [
    "ModelFactory",
    "ModelManager",
    "get_model_manager",
    "get_model",
    "get_embedding_model",
    "SUPPORTED_BACKENDS",
]
