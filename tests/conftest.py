"""
Configuración y fixtures compartidos para tests.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Añadir el directorio raíz al path para imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


# =============================================================================
# Configuración de pytest-asyncio
# =============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Crea un event loop para toda la sesión de tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Fixtures para mocking de respuestas de modelos
# =============================================================================

@pytest.fixture
def mock_ollama_response() -> dict[str, Any]:
    """Respuesta típica de Ollama."""
    return {
        "model": "qwen2.5:14b",
        "message": {
            "role": "assistant",
            "content": "Esta es una respuesta de prueba del modelo."
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 50,
        "eval_count": 25,
    }


@pytest.fixture
def mock_openai_response() -> dict[str, Any]:
    """Respuesta típica de OpenAI API compatible."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Esta es una respuesta de prueba."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 25,
            "total_tokens": 75
        }
    }


@pytest.fixture
def mock_embedding_response() -> dict[str, Any]:
    """Respuesta típica de embeddings."""
    return {
        "embeddings": [[0.1, 0.2, 0.3] * 256],  # 768 dimensiones
        "prompt_eval_count": 10,
    }


@pytest.fixture
def sample_messages() -> list[dict[str, str]]:
    """Mensajes de ejemplo para pruebas."""
    return [
        {"role": "system", "content": "Eres un asistente útil."},
        {"role": "user", "content": "Hola, ¿cómo estás?"},
    ]


@pytest.fixture
def sample_math_problem() -> str:
    """Problema matemático de ejemplo."""
    return "Un tren sale de Madrid a las 8:00 a 120 km/h. Otro tren sale de Barcelona (600 km) a las 9:00 a 150 km/h hacia Madrid. ¿A qué hora se cruzan?"


# =============================================================================
# Fixtures para mocking de HTTP clients
# =============================================================================

@pytest.fixture
def mock_httpx_client() -> Generator[MagicMock, None, None]:
    """Mock del cliente httpx para tests sin red."""
    with patch("httpx.AsyncClient") as mock:
        client_instance = AsyncMock()
        mock.return_value = client_instance
        mock.return_value.__aenter__ = AsyncMock(return_value=client_instance)
        mock.return_value.__aexit__ = AsyncMock(return_value=None)
        yield client_instance


# =============================================================================
# Fixtures para configuración
# =============================================================================

@pytest.fixture
def mock_settings() -> Generator[MagicMock, None, None]:
    """Mock de settings para tests."""
    with patch("config.settings.get_settings") as mock:
        settings = MagicMock()
        settings.ollama.base_url = "http://localhost:11434"
        settings.ollama.timeout = 60
        settings.openai_local.base_url = "http://localhost:8000/v1"
        settings.openai_local.api_key = "not-needed"
        settings.openai_local.timeout = 60
        settings.huggingface.cache_dir = "/tmp/hf_cache"
        settings.huggingface.device_map = "auto"
        settings.huggingface.torch_dtype = "bfloat16"
        settings.huggingface.load_in_4bit = False
        settings.huggingface.load_in_8bit = False
        mock.return_value = settings
        yield settings


# =============================================================================
# Fixtures para tests de integración (requieren servicios reales)
# =============================================================================

@pytest.fixture
def ollama_available() -> bool:
    """Verifica si Ollama está disponible para tests de integración."""
    import httpx
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


@pytest.fixture
def skip_if_no_ollama(ollama_available: bool) -> None:
    """Skip test si Ollama no está disponible."""
    if not ollama_available:
        pytest.skip("Ollama no disponible - test de integración omitido")


# =============================================================================
# Markers personalizados
# =============================================================================

def pytest_configure(config: Any) -> None:
    """Configura markers personalizados."""
    config.addinivalue_line(
        "markers",
        "integration: test de integración que requiere servicios externos"
    )
    config.addinivalue_line(
        "markers",
        "slow: test lento que puede omitirse con --skip-slow"
    )
    config.addinivalue_line(
        "markers",
        "gpu: test que requiere GPU"
    )


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Modifica la colección de tests según opciones."""
    # Skip tests lentos si se especifica --skip-slow
    if config.getoption("--skip-slow", default=False):
        skip_slow = pytest.mark.skip(reason="--skip-slow especificado")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_addoption(parser: Any) -> None:
    """Añade opciones de línea de comandos."""
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="Omitir tests marcados como lentos"
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Ejecutar tests de integración"
    )
