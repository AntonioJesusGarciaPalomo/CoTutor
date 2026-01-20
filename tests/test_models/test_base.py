"""
Tests para las clases base de adaptadores de modelos.
"""

from __future__ import annotations

import pytest

from src.core.types import Message, MessageRole, ModelResponse
from src.models.base import BaseModelAdapter


class ConcreteAdapter(BaseModelAdapter):
    """Implementación concreta para testing."""
    
    async def generate(self, messages, **kwargs):
        content = f"Generated from {len(messages)} messages"
        return self._create_response(
            content=content,
            prompt_tokens=10,
            completion_tokens=5,
        )
    
    async def generate_stream(self, messages, **kwargs):
        yield "Hello "
        yield "World"
    
    async def health_check(self):
        return True
    
    async def get_model_info(self):
        return {"model": self.model_name, "backend": self.backend_name}


class TestBaseModelAdapter:
    """Tests para BaseModelAdapter."""
    
    def test_init(self) -> None:
        """Test de inicialización básica."""
        adapter = ConcreteAdapter(
            model_name="test-model",
            backend_name="test-backend",
        )
        
        assert adapter.model_name == "test-model"
        assert adapter.backend_name == "test-backend"
        assert adapter.model_id == "test-backend/test-model"
        assert adapter.is_loaded is False
    
    def test_model_id_property(self) -> None:
        """Test del property model_id."""
        adapter = ConcreteAdapter("llama3.1:8b", "ollama")
        assert adapter.model_id == "ollama/llama3.1:8b"
    
    def test_normalize_messages_from_dict(self) -> None:
        """Test de normalización de mensajes desde diccionarios."""
        adapter = ConcreteAdapter("test", "test")
        
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
        ]
        
        normalized = adapter._normalize_messages(messages)
        
        assert len(normalized) == 2
        assert normalized[0]["role"] == "system"
        assert normalized[0]["content"] == "System prompt"
        assert normalized[1]["role"] == "user"
    
    def test_normalize_messages_from_message_objects(self) -> None:
        """Test de normalización de objetos Message."""
        adapter = ConcreteAdapter("test", "test")
        
        messages = [
            Message(role=MessageRole.SYSTEM, content="System"),
            Message(role=MessageRole.USER, content="User"),
        ]
        
        normalized = adapter._normalize_messages(messages)
        
        assert len(normalized) == 2
        assert normalized[0]["role"] == "system"
        assert normalized[1]["role"] == "user"
    
    def test_normalize_messages_mixed(self) -> None:
        """Test de normalización con tipos mixtos."""
        adapter = ConcreteAdapter("test", "test")
        
        messages = [
            {"role": "system", "content": "System"},
            Message(role=MessageRole.USER, content="User"),
        ]
        
        normalized = adapter._normalize_messages(messages)
        
        assert len(normalized) == 2
        assert all(isinstance(m, dict) for m in normalized)
    
    def test_create_response(self) -> None:
        """Test de creación de ModelResponse."""
        adapter = ConcreteAdapter("test", "test")
        
        response = adapter._create_response(
            content="Test content",
            prompt_tokens=100,
            completion_tokens=50,
            generation_time_ms=1000,
            finish_reason="stop",
        )
        
        assert isinstance(response, ModelResponse)
        assert response.content == "Test content"
        assert response.model == "test/test"
        assert response.prompt_tokens == 100
        assert response.completion_tokens == 50
        assert response.total_tokens == 150
        assert response.generation_time_ms == 1000
        assert response.tokens_per_second == 50.0  # 50 tokens / 1 segundo
        assert response.finish_reason == "stop"
    
    def test_create_response_partial_metrics(self) -> None:
        """Test de creación con métricas parciales."""
        adapter = ConcreteAdapter("test", "test")
        
        response = adapter._create_response(
            content="Test",
            prompt_tokens=100,
            # Sin completion_tokens
        )
        
        assert response.prompt_tokens == 100
        assert response.completion_tokens is None
        assert response.total_tokens is None
        assert response.tokens_per_second is None
    
    @pytest.mark.asyncio
    async def test_generate(self) -> None:
        """Test de generación."""
        adapter = ConcreteAdapter("test", "test")
        
        messages = [{"role": "user", "content": "Hello"}]
        response = await adapter.generate(messages)
        
        assert response.content == "Generated from 1 messages"
        assert response.model == "test/test"
    
    @pytest.mark.asyncio
    async def test_generate_stream(self) -> None:
        """Test de generación con streaming."""
        adapter = ConcreteAdapter("test", "test")
        
        messages = [{"role": "user", "content": "Hello"}]
        chunks = []
        
        async for chunk in adapter.generate_stream(messages):
            chunks.append(chunk)
        
        assert chunks == ["Hello ", "World"]
    
    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        """Test de health check."""
        adapter = ConcreteAdapter("test", "test")
        
        is_healthy = await adapter.health_check()
        
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_count_tokens_default(self) -> None:
        """Test de conteo de tokens (aproximación por defecto)."""
        adapter = ConcreteAdapter("test", "test")
        
        # 40 caracteres ÷ 4 = 10 tokens aproximados
        text = "Esta es una prueba de cuarenta caractere"
        tokens = await adapter.count_tokens(text)
        
        assert tokens == 10
    
    @pytest.mark.asyncio
    async def test_load_unload(self) -> None:
        """Test de carga y descarga."""
        adapter = ConcreteAdapter("test", "test")
        
        assert adapter.is_loaded is False
        
        await adapter.load()
        assert adapter.is_loaded is True
        
        await adapter.unload()
        assert adapter.is_loaded is False
    
    @pytest.mark.asyncio
    async def test_embed_not_implemented(self) -> None:
        """Test de que embed lanza NotImplementedError por defecto."""
        adapter = ConcreteAdapter("test", "test")
        
        with pytest.raises(NotImplementedError):
            await adapter.embed(["test"])
    
    def test_repr(self) -> None:
        """Test de representación string."""
        adapter = ConcreteAdapter("test-model", "test-backend")
        
        repr_str = repr(adapter)
        
        assert "ConcreteAdapter" in repr_str
        assert "test-backend/test-model" in repr_str
        assert "loaded=False" in repr_str
