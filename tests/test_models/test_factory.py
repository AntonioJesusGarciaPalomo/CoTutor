"""
Tests para ModelFactory y ModelManager.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import BackendNotSupportedError
from src.models.factory import (
    ModelFactory,
    ModelManager,
    get_model,
    get_model_manager,
)
from src.models.huggingface_local import HuggingFaceLocalAdapter
from src.models.ollama_adapter import OllamaAdapter, OllamaEmbeddingAdapter
from src.models.openai_local import OpenAILocalAdapter


class TestModelFactory:
    """Tests para ModelFactory."""
    
    def setup_method(self) -> None:
        """Limpia la caché antes de cada test."""
        ModelFactory.clear_cache()
    
    def test_create_ollama_adapter(self, mock_settings: MagicMock) -> None:
        """Test de creación de adaptador Ollama."""
        adapter = ModelFactory.create("ollama/llama3.1:8b")
        
        assert isinstance(adapter, OllamaAdapter)
        assert adapter.model_name == "llama3.1:8b"
        assert adapter.backend_name == "ollama"
    
    def test_create_openai_local_adapter(self, mock_settings: MagicMock) -> None:
        """Test de creación de adaptador OpenAI Local."""
        adapter = ModelFactory.create("openai_local/meta-llama/Llama-3.1-8B")
        
        assert isinstance(adapter, OpenAILocalAdapter)
        assert adapter.model_name == "meta-llama/Llama-3.1-8B"
        assert adapter.backend_name == "openai_local"
    
    def test_create_huggingface_adapter(self, mock_settings: MagicMock) -> None:
        """Test de creación de adaptador HuggingFace."""
        adapter = ModelFactory.create("huggingface/Qwen/Qwen2.5-14B-Instruct")
        
        assert isinstance(adapter, HuggingFaceLocalAdapter)
        assert adapter.model_name == "Qwen/Qwen2.5-14B-Instruct"
        assert adapter.backend_name == "huggingface"
    
    def test_create_with_custom_config(self, mock_settings: MagicMock) -> None:
        """Test de creación con configuración personalizada."""
        adapter = ModelFactory.create(
            "ollama/qwen2.5:14b",
            base_url="http://192.168.1.100:11434",
            timeout=120,
        )
        
        assert adapter.base_url == "http://192.168.1.100:11434"
        assert adapter.timeout == 120
    
    def test_create_invalid_model_id_no_slash(self) -> None:
        """Test de error con model_id sin /."""
        with pytest.raises(ValueError) as exc_info:
            ModelFactory.create("llama3.1:8b")
        
        assert "Formato de model_id inválido" in str(exc_info.value)
    
    def test_create_unsupported_backend(self, mock_settings: MagicMock) -> None:
        """Test de error con backend no soportado."""
        with pytest.raises(BackendNotSupportedError) as exc_info:
            ModelFactory.create("unsupported/model")
        
        assert "unsupported" in str(exc_info.value)
    
    def test_cache_returns_same_instance(self, mock_settings: MagicMock) -> None:
        """Test de que la caché retorna la misma instancia."""
        adapter1 = ModelFactory.create("ollama/llama3.1:8b", use_cache=True)
        adapter2 = ModelFactory.create("ollama/llama3.1:8b", use_cache=True)
        
        assert adapter1 is adapter2
    
    def test_cache_disabled_returns_new_instance(self, mock_settings: MagicMock) -> None:
        """Test de que sin caché retorna nueva instancia."""
        adapter1 = ModelFactory.create("ollama/llama3.1:8b", use_cache=False)
        adapter2 = ModelFactory.create("ollama/llama3.1:8b", use_cache=False)
        
        assert adapter1 is not adapter2
    
    def test_create_embedding_ollama(self, mock_settings: MagicMock) -> None:
        """Test de creación de adaptador de embeddings Ollama."""
        adapter = ModelFactory.create_embedding("ollama/nomic-embed-text")
        
        assert isinstance(adapter, OllamaEmbeddingAdapter)
        assert adapter.model_name == "nomic-embed-text"
    
    def test_clear_cache(self, mock_settings: MagicMock) -> None:
        """Test de limpieza de caché."""
        ModelFactory.create("ollama/llama3.1:8b", use_cache=True)
        assert len(ModelFactory.list_cached()) > 0
        
        ModelFactory.clear_cache()
        
        assert len(ModelFactory.list_cached()) == 0


class TestModelManager:
    """Tests para ModelManager."""
    
    @pytest.fixture
    def manager(self) -> ModelManager:
        """Crea un gestor de modelos."""
        return ModelManager()
    
    @pytest.mark.asyncio
    async def test_get_creates_and_caches(
        self,
        manager: ModelManager,
        mock_settings: MagicMock,
    ) -> None:
        """Test de que get crea y cachea el adaptador."""
        with patch.object(OllamaAdapter, "health_check", return_value=True):
            adapter1 = await manager.get("ollama/llama3.1:8b")
            adapter2 = await manager.get("ollama/llama3.1:8b")
        
        assert adapter1 is adapter2
        assert "ollama/llama3.1:8b" in manager
    
    def test_contains(
        self,
        manager: ModelManager,
    ) -> None:
        """Test del operador in."""
        assert "ollama/model1" not in manager


class TestConvenienceFunctions:
    """Tests para funciones de conveniencia."""
    
    def test_get_model_manager_singleton(self) -> None:
        """Test de que get_model_manager retorna singleton."""
        manager1 = get_model_manager()
        manager2 = get_model_manager()
        
        assert manager1 is manager2
