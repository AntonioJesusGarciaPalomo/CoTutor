#!/usr/bin/env python3
"""
Ejemplo de uso de la capa de abstracción de modelos.

Este script demuestra cómo usar los diferentes backends
(Ollama, OpenAI Local, HuggingFace) de forma intercambiable.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    ModelFactory,
    ModelManager,
    get_model,
    get_model_manager,
    SUPPORTED_BACKENDS,
)


async def example_basic_generation():
    """Ejemplo básico de generación de texto."""
    print("\n" + "=" * 60)
    print("EJEMPLO 1: Generación básica con Ollama")
    print("=" * 60)
    
    # Crear adaptador usando el Factory
    adapter = ModelFactory.create("ollama/llama3.1:8b")
    
    # Verificar disponibilidad
    if not await adapter.health_check():
        print("❌ Ollama no está disponible. Asegúrate de que esté corriendo.")
        print("   Ejecuta: ollama serve")
        return
    
    print(f"✅ Conectado a: {adapter.model_id}")
    
    # Generar respuesta
    print("\n📝 Generando respuesta...")
    response = await adapter.generate(
        messages=[
            {"role": "system", "content": "Eres un asistente educativo amable."},
            {"role": "user", "content": "¿Qué es un número primo? Explícalo brevemente."},
        ],
        temperature=0.7,
        max_tokens=200,
    )
    
    print(f"\n💬 Respuesta:\n{response.content}")
    print(f"\n📊 Métricas:")
    print(f"   - Tokens prompt: {response.prompt_tokens}")
    print(f"   - Tokens generados: {response.completion_tokens}")
    print(f"   - Tiempo: {response.generation_time_ms:.0f}ms")
    if response.tokens_per_second:
        print(f"   - Velocidad: {response.tokens_per_second:.1f} tokens/s")


async def example_streaming():
    """Ejemplo de generación con streaming."""
    print("\n" + "=" * 60)
    print("EJEMPLO 2: Generación con streaming")
    print("=" * 60)
    
    adapter = ModelFactory.create("ollama/llama3.1:8b")
    
    if not await adapter.health_check():
        print("❌ Ollama no está disponible")
        return
    
    print("\n📝 Generando con streaming...\n")
    print("💬 Respuesta: ", end="", flush=True)
    
    async for chunk in adapter.generate_stream(
        messages=[
            {"role": "user", "content": "Cuenta del 1 al 5 con una palabra por línea."},
        ],
        max_tokens=50,
    ):
        print(chunk, end="", flush=True)
    
    print("\n")


async def example_multiple_backends():
    """Ejemplo de uso de múltiples backends."""
    print("\n" + "=" * 60)
    print("EJEMPLO 3: Múltiples backends")
    print("=" * 60)
    
    print(f"\n🔌 Backends soportados: {', '.join(SUPPORTED_BACKENDS)}")
    
    # Ollama
    ollama = ModelFactory.create("ollama/llama3.1:8b")
    ollama_ok = await ollama.health_check()
    print(f"\n   Ollama: {'✅ Disponible' if ollama_ok else '❌ No disponible'}")
    
    # OpenAI Local (vLLM)
    vllm = ModelFactory.create(
        "openai_local/default",
        base_url="http://localhost:8000/v1",
    )
    vllm_ok = await vllm.health_check()
    print(f"   vLLM: {'✅ Disponible' if vllm_ok else '❌ No disponible'}")
    
    # Usar el que esté disponible
    if ollama_ok:
        adapter = ollama
    elif vllm_ok:
        adapter = vllm
    else:
        print("\n❌ No hay ningún backend disponible")
        return
    
    response = await adapter.generate([
        {"role": "user", "content": "Di 'Hola' en tres idiomas diferentes."}
    ], max_tokens=50)
    
    print(f"\n💬 Usando {adapter.model_id}:")
    print(f"   {response.content}")


async def example_model_manager():
    """Ejemplo de uso del ModelManager."""
    print("\n" + "=" * 60)
    print("EJEMPLO 4: Usando ModelManager")
    print("=" * 60)
    
    manager = get_model_manager()
    
    # Precargar modelos
    print("\n📦 Precargando modelos...")
    results = await manager.preload([
        "ollama/llama3.1:8b",
    ], verify=True)
    
    for model_id, success in results.items():
        status = "✅ OK" if success else "❌ Error"
        print(f"   {model_id}: {status}")
    
    # Listar modelos cargados
    loaded = manager.list_loaded()
    print(f"\n📋 Modelos cargados: {loaded}")
    
    # Usar modelo del manager
    if loaded:
        adapter = await manager.get(loaded[0])
        response = await adapter.generate([
            {"role": "user", "content": "¿Cuánto es 7 × 8?"}
        ], max_tokens=20)
        print(f"\n💬 Respuesta: {response.content}")
    
    # Limpiar
    await manager.cleanup()
    print("\n🧹 Modelos descargados")


async def example_embeddings():
    """Ejemplo de generación de embeddings."""
    print("\n" + "=" * 60)
    print("EJEMPLO 5: Generación de embeddings")
    print("=" * 60)
    
    embedder = ModelFactory.create_embedding("ollama/nomic-embed-text")
    
    if not await embedder.health_check():
        print("❌ Modelo de embeddings no disponible")
        print("   Ejecuta: ollama pull nomic-embed-text")
        return
    
    texts = [
        "La derivada de x² es 2x",
        "El teorema de Pitágoras dice que a² + b² = c²",
        "Python es un lenguaje de programación",
    ]
    
    print(f"\n📝 Generando embeddings para {len(texts)} textos...")
    
    result = await embedder.embed(texts)
    
    print(f"\n📊 Resultado:")
    print(f"   - Dimensiones: {result.dimensions}")
    print(f"   - Vectores generados: {len(result.embeddings)}")
    if result.generation_time_ms:
        print(f"   - Tiempo: {result.generation_time_ms:.0f}ms")
    
    # Calcular similitud simple (coseno)
    import numpy as np
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    emb = result.embeddings
    print(f"\n🔍 Similitud entre textos:")
    print(f"   Texto 1 vs Texto 2: {cosine_similarity(emb[0], emb[1]):.3f}")
    print(f"   Texto 1 vs Texto 3: {cosine_similarity(emb[0], emb[2]):.3f}")
    print(f"   Texto 2 vs Texto 3: {cosine_similarity(emb[1], emb[2]):.3f}")


async def example_model_info():
    """Ejemplo de obtención de información del modelo."""
    print("\n" + "=" * 60)
    print("EJEMPLO 6: Información del modelo")
    print("=" * 60)
    
    adapter = ModelFactory.create("ollama/llama3.1:8b")
    
    if not await adapter.health_check():
        print("❌ Modelo no disponible")
        return
    
    info = await adapter.get_model_info()
    
    print(f"\n📋 Información de {adapter.model_id}:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Listar todos los modelos disponibles
    if hasattr(adapter, "list_models"):
        models = await adapter.list_models()
        print(f"\n📦 Modelos disponibles en Ollama:")
        for model in models[:5]:  # Solo los primeros 5
            print(f"   - {model.get('name', 'unknown')}")


async def main():
    """Ejecuta todos los ejemplos."""
    print("\n" + "=" * 60)
    print(" AULA AI TUTOR - Ejemplos de la Capa de Abstracción de Modelos")
    print("=" * 60)
    
    try:
        await example_basic_generation()
        await example_streaming()
        await example_multiple_backends()
        await example_model_manager()
        await example_embeddings()
        await example_model_info()
        
        print("\n" + "=" * 60)
        print(" ✅ Todos los ejemplos completados")
        print("=" * 60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrumpido por el usuario")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
