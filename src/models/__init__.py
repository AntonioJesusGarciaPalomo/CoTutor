"""
Módulo de abstracción de modelos del sistema Aula AI Tutor.

Este módulo proporciona una capa de abstracción unificada para trabajar
con diferentes backends de modelos de lenguaje:

- **Ollama**: Modelos locales via servidor Ollama
- **OpenAI Local**: Servidores compatibles con API OpenAI (vLLM, llama.cpp, LM Studio)
- **HuggingFace**: Modelos cargados directamente desde HuggingFace Hub

Ejemplo de uso básico:
    ```python
    from src.models import get_model, ModelFactory
    
    # Usando el gestor global (recomendado)
    model = await get_model("ollama/qwen2.5:14b")
    response = await model.generate([
        {"role": "user", "content": "Hola, ¿cómo estás?"}
    ])
    print(response.content)
    
    # Usando la factory directamente
    adapter = ModelFactory.create("huggingface/Qwen/Qwen2.5-14B-Instruct")
    await adapter.load()
    response = await adapter.generate([...])
    ```

El formato de identificador de modelo es: "backend/model_name"
- ollama/llama3.1:8b
- openai_local/meta-llama/Llama-3.1-8B-Instruct  
- huggingface/Qwen/Qwen2.5-14B-Instruct
"""

from src.models.base import (
    BaseEmbeddingAdapter,
    BaseModelAdapter,
    EmbeddingAdapterFactory,
    ModelAdapterFactory,
)
from src.models.factory import (
    SUPPORTED_BACKENDS,
    ModelFactory,
    ModelManager,
    get_embedding_model,
    get_model,
    get_model_manager,
)
from src.models.huggingface_local import (
    HuggingFaceEmbeddingAdapter,
    HuggingFaceLocalAdapter,
)
from src.models.ollama_adapter import OllamaAdapter, OllamaEmbeddingAdapter
from src.models.openai_local import OpenAILocalAdapter, OpenAILocalEmbeddingAdapter

__all__ = [
    # Base classes
    "BaseModelAdapter",
    "BaseEmbeddingAdapter",
    "ModelAdapterFactory",
    "EmbeddingAdapterFactory",
    # Ollama
    "OllamaAdapter",
    "OllamaEmbeddingAdapter",
    # OpenAI Local
    "OpenAILocalAdapter",
    "OpenAILocalEmbeddingAdapter",
    # HuggingFace
    "HuggingFaceLocalAdapter",
    "HuggingFaceEmbeddingAdapter",
    # Factory
    "ModelFactory",
    "ModelManager",
    "get_model_manager",
    "get_model",
    "get_embedding_model",
    "SUPPORTED_BACKENDS",
]
