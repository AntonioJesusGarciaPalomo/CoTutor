"""
Tests para el adaptador de servidores compatibles con OpenAI API.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import (
    ModelConnectionError,
    ModelGenerationError,
    ModelNotFoundError,
    ModelTimeoutError,
)
from src.core.types import EmbeddingResponse, ModelResponse
from src.models.openai_local import OpenAILocalAdapter, OpenAILocalEmbeddingAdapter


class TestOpenAILocalAdapter:
    """Tests para OpenAILocalAdapter."""
    
    def test_initialization(self):
        """Test de inicialización del adaptador."""
        adapter = OpenAILocalAdapter(
            model_name="llama-3.1-8b",
            base_url="http://localhost:8000/v1",
            api_key="test-key",
            timeout=60.0,
            server_type="vllm",
        )
        
        assert adapter.model_name == "llama-3.1-8b"
        assert adapter.backend_name == "openai_local"
        assert adapter.base_url == "http://localhost:8000/v1"
        assert adapter.api_key == "test-key"
        assert adapter.timeout == 60.0
        assert adapter.server_type == "vllm"
        assert adapter.model_id == "openai_local/llama-3.1-8b"
    
    def test_default_api_key(self):
        """Test de API key por defecto."""
        adapter = OpenAILocalAdapter(model_name="test")
        
        assert adapter.api_key == "not-needed"
    
    @pytest.mark.asyncio
    async def test_generate_success(self, sample_messages, mock_openai_response):
        """Test de generación exitosa."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.is_closed = False
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openai_response
            mock_response.raise_for_status = MagicMock()
            
            mock_client.post.return_value = mock_response
            
            adapter = OpenAILocalAdapter(model_name="llama-3.1-8b")
            response = await adapter.generate(sample_messages)
            
            assert isinstance(response, ModelResponse)
            assert response.content == "Esta es una respuesta de prueba."
            assert response.model == "openai_local/llama-3.1-8b"
            assert response.prompt_tokens == 50
            assert response.completion_tokens == 25
            assert response.finish_reason == "stop"
    
    @pytest.mark.asyncio
    async def test_generate_with_parameters(self, mock_openai_response):
        """Test de generación con parámetros personalizados."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.is_closed = False
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openai_response
            mock_response.raise_for_status = MagicMock()
            
            mock_client.post.return_value = mock_response
            
            adapter = OpenAILocalAdapter(model_name="test")
            
            await adapter.generate(
                [{"role": "user", "content": "test"}],
                temperature=0.5,
                max_tokens=100,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1,
            )
            
            # Verificar payload
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json", {})
            
            assert payload["temperature"] == 0.5
            assert payload["max_tokens"] == 100
            assert payload["top_p"] == 0.9
            assert payload["frequency_penalty"] == 0.1
            assert payload["presence_penalty"] == 0.1
    
    @pytest.mark.asyncio
    async def test_generate_model_not_found(self, sample_messages):
        """Test de error cuando el modelo no existe."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.is_closed = False
            
            mock_response = MagicMock()
            mock_response.status_code = 404
            
            mock_client.post.return_value = mock_response
            
            adapter = OpenAILocalAdapter(model_name="modelo-inexistente")
            
            with pytest.raises(ModelNotFoundError):
                await adapter.generate(sample_messages)
    
    @pytest.mark.asyncio
    async def test_generate_bad_request(self, sample_messages):
        """Test de error 400."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.is_closed = False
            
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.json.return_value = {
                "error": {"message": "Invalid request"}
            }
            
            mock_client.post.return_value = mock_response
            
            adapter = OpenAILocalAdapter(model_name="test")
            
            with pytest.raises(ModelGenerationError) as exc_info:
                await adapter.generate(sample_messages)
            
            assert "Invalid request" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generate_no_choices(self, sample_messages):
        """Test de error cuando no hay choices."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.is_closed = False
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"choices": []}
            mock_response.raise_for_status = MagicMock()
            
            mock_client.post.return_value = mock_response
            
            adapter = OpenAILocalAdapter(model_name="test")
            
            with pytest.raises(ModelGenerationError) as exc_info:
                await adapter.generate(sample_messages)
            
            assert "No se recibieron choices" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generate_connection_error(self, sample_messages):
        """Test de error de conexión."""
        import httpx
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.is_closed = False
            
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            
            adapter = OpenAILocalAdapter(model_name="test")
            
            with pytest.raises(ModelConnectionError):
                await adapter.generate(sample_messages)
    
    @pytest.mark.asyncio
    async def test_generate_timeout(self, sample_messages):
        """Test de timeout."""
        import httpx
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.is_closed = False
            
            mock_client.post.side_effect = httpx.TimeoutException("Timeout")
            
            adapter = OpenAILocalAdapter(model_name="test", timeout=5.0)
            
            with pytest.raises(ModelTimeoutError):
                await adapter.generate(sample_messages)
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test de health check exitoso."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.is_closed = False
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            
            mock_client.get.return_value = mock_response
            
            adapter = OpenAILocalAdapter(model_name="test")
            result = await adapter.health_check()
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_models_not_found(self):
        """Test de health check cuando /models no existe pero el servidor responde."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.is_closed = False
            
            # GET /models devuelve 404
            mock_get_response = MagicMock()
            mock_get_response.status_code = 404
            mock_client.get.return_value = mock_get_response
            
            # POST /chat/completions funciona
            mock_post_response = MagicMock()
            mock_post_response.status_code = 200
            mock_client.post.return_value = mock_post_response
            
            adapter = OpenAILocalAdapter(model_name="test")
            result = await adapter.health_check()
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test de listado de modelos."""
        models_response = {
            "data": [
                {"id": "llama-3.1-8b", "owned_by": "meta"},
                {"id": "qwen2.5-14b", "owned_by": "alibaba"},
            ]
        }
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.is_closed = False
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = models_response
            mock_response.raise_for_status = MagicMock()
            
            mock_client.get.return_value = mock_response
            
            adapter = OpenAILocalAdapter(model_name="test")
            models = await adapter.list_models()
            
            assert len(models) == 2
            assert models[0]["id"] == "llama-3.1-8b"
    
    @pytest.mark.asyncio
    async def test_embed_success(self):
        """Test de generación de embeddings."""
        embed_response = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ],
            "usage": {"total_tokens": 10},
        }
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.is_closed = False
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = embed_response
            mock_response.raise_for_status = MagicMock()
            
            mock_client.post.return_value = mock_response
            
            adapter = OpenAILocalAdapter(
                model_name="text-embedding-model",
                server_type="vllm",
            )
            result = await adapter.embed(["text1", "text2"])
            
            assert isinstance(result, EmbeddingResponse)
            assert len(result.embeddings) == 2
            assert result.dimensions == 3
            assert result.total_tokens == 10
    
    @pytest.mark.asyncio
    async def test_get_model_info(self):
        """Test de obtención de información del modelo."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.is_closed = False
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"id": "my-model", "owned_by": "local", "created": 1234567890}
                ]
            }
            
            mock_client.get.return_value = mock_response
            
            adapter = OpenAILocalAdapter(model_name="my-model", server_type="vllm")
            info = await adapter.get_model_info()
            
            assert info["model"] == "my-model"
            assert info["backend"] == "openai_local"
            assert info["server_type"] == "vllm"


class TestOpenAILocalEmbeddingAdapter:
    """Tests para OpenAILocalEmbeddingAdapter."""
    
    def test_initialization(self):
        """Test de inicialización."""
        adapter = OpenAILocalEmbeddingAdapter(
            model_name="bge-large",
            base_url="http://localhost:8000/v1",
            dimensions=1024,
        )
        
        assert adapter.model_name == "bge-large"
        assert adapter.backend_name == "openai_local"
        assert adapter.dimensions == 1024
    
    @pytest.mark.asyncio
    async def test_embed(self):
        """Test de embedding."""
        embed_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "usage": {"total_tokens": 5},
        }
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.is_closed = False
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = embed_response
            mock_response.raise_for_status = MagicMock()
            
            mock_client.post.return_value = mock_response
            
            adapter = OpenAILocalEmbeddingAdapter(model_name="embed")
            result = await adapter.embed(["test"])
            
            assert len(result.embeddings) == 1
            assert result.embeddings[0] == [0.1, 0.2, 0.3]


@pytest.mark.integration
@pytest.mark.vllm
class TestOpenAILocalAdapterIntegration:
    """
    Tests de integración que requieren vLLM u otro servidor corriendo.
    
    Ejecutar con: pytest -m vllm
    """
    
    @pytest.mark.asyncio
    async def test_real_generation(self):
        """Test de generación real con vLLM."""
        adapter = OpenAILocalAdapter(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            base_url="http://localhost:8000/v1",
        )
        
        if not await adapter.health_check():
            pytest.skip("vLLM no está disponible")
        
        response = await adapter.generate([
            {"role": "user", "content": "Di 'hola' y nada más."}
        ], max_tokens=10)
        
        assert response.content is not None
        assert len(response.content) > 0
