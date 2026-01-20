"""
Tests para el adaptador de Ollama.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.core.exceptions import (
    ModelConnectionError,
    ModelNotFoundError,
    ModelTimeoutError,
)
from src.models.ollama_adapter import OllamaAdapter, OllamaEmbeddingAdapter


class TestOllamaAdapter:
    """Tests para OllamaAdapter."""
    
    @pytest.fixture
    def adapter(self) -> OllamaAdapter:
        """Crea un adaptador de prueba."""
        return OllamaAdapter(
            model_name="llama3.1:8b",
            base_url="http://localhost:11434",
            timeout=60,
        )
    
    def test_init(self, adapter: OllamaAdapter) -> None:
        """Test de inicialización."""
        assert adapter.model_name == "llama3.1:8b"
        assert adapter.backend_name == "ollama"
        assert adapter.base_url == "http://localhost:11434"
        assert adapter.timeout == 60
        assert adapter.supports_streaming is True
        assert adapter.supports_embeddings is True
    
    def test_model_id(self, adapter: OllamaAdapter) -> None:
        """Test del identificador de modelo."""
        assert adapter.model_id == "ollama/llama3.1:8b"
    
    @pytest.mark.asyncio
    async def test_generate_success(
        self,
        adapter: OllamaAdapter,
        mock_ollama_response: dict[str, Any],
    ) -> None:
        """Test de generación exitosa."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_ollama_response
        mock_response.raise_for_status = MagicMock()
        
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            response = await adapter.generate([
                {"role": "user", "content": "Hola"}
            ])
        
        assert response.content == "Esta es una respuesta de prueba del modelo."
        assert response.model == "ollama/llama3.1:8b"
        assert response.prompt_tokens == 50
        assert response.completion_tokens == 25
    
    @pytest.mark.asyncio
    async def test_generate_model_not_found(
        self,
        adapter: OllamaAdapter,
    ) -> None:
        """Test de error cuando el modelo no existe."""
        mock_response = AsyncMock()
        mock_response.status_code = 404
        
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            with pytest.raises(ModelNotFoundError) as exc_info:
                await adapter.generate([{"role": "user", "content": "Hola"}])
            
            assert "llama3.1:8b" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generate_connection_error(
        self,
        adapter: OllamaAdapter,
    ) -> None:
        """Test de error de conexión."""
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_get_client.return_value = mock_client
            
            with pytest.raises(ModelConnectionError) as exc_info:
                await adapter.generate([{"role": "user", "content": "Hola"}])
            
            assert "ollama" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generate_timeout_error(
        self,
        adapter: OllamaAdapter,
    ) -> None:
        """Test de error de timeout."""
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Timeout")
            mock_get_client.return_value = mock_client
            
            with pytest.raises(ModelTimeoutError):
                await adapter.generate([{"role": "user", "content": "Hola"}])
    
    @pytest.mark.asyncio
    async def test_health_check_success(
        self,
        adapter: OllamaAdapter,
    ) -> None:
        """Test de health check exitoso."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            is_healthy = await adapter.health_check()
        
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(
        self,
        adapter: OllamaAdapter,
    ) -> None:
        """Test de health check fallido."""
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection failed")
            mock_get_client.return_value = mock_client
            
            is_healthy = await adapter.health_check()
        
        assert is_healthy is False


class TestOllamaEmbeddingAdapter:
    """Tests para OllamaEmbeddingAdapter."""
    
    @pytest.fixture
    def adapter(self) -> OllamaEmbeddingAdapter:
        """Crea un adaptador de embeddings de prueba."""
        return OllamaEmbeddingAdapter(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",
        )
    
    def test_init(self, adapter: OllamaEmbeddingAdapter) -> None:
        """Test de inicialización."""
        assert adapter.model_name == "nomic-embed-text"
        assert adapter.backend_name == "ollama"
        assert adapter.model_id == "ollama/nomic-embed-text"


# =============================================================================
# Tests de integración (requieren Ollama corriendo)
# =============================================================================

@pytest.mark.integration
class TestOllamaIntegration:
    """Tests de integración con Ollama real."""
    
    @pytest.fixture
    def adapter(self) -> OllamaAdapter:
        """Crea un adaptador apuntando a Ollama real."""
        return OllamaAdapter(
            model_name="llama3.2:1b",
            base_url="http://localhost:11434",
        )
    
    @pytest.mark.asyncio
    async def test_real_health_check(
        self,
        adapter: OllamaAdapter,
        skip_if_no_ollama: None,
    ) -> None:
        """Test de health check real."""
        is_healthy = await adapter.health_check()
        assert is_healthy is True
