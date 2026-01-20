#!/usr/bin/env python3
"""
Ejemplo de uso de la capa de abstracción de modelos.

Este script demuestra cómo usar los diferentes backends
(Ollama, OpenAI Local, HuggingFace) de forma transparente.

Requisitos:
- Ollama corriendo en localhost:11434
- Modelo llama3.2:1b descargado (ollama pull llama3.2:1b)

Ejecutar:
    python examples/model_usage_example.py
"""

import asyncio
import sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ModelFactory, get_model, get_model_manager


async def basic_generation_example():
    """Ejemplo básico de generación de texto."""
    print("\n" + "=" * 60)
    print("EJEMPLO 1: Generación Básica con Ollama")
    print("=" * 60)
    
    try:
        # Crear adaptador usando la factory
        model = ModelFactory.create("ollama/llama3.2:1b")
        
        # Verificar conexión
        is_healthy = await model.health_check()
        if not is_healthy:
            print("❌ Ollama no está disponible. Asegúrate de que está corriendo.")
            return
        
        print("✅ Conexión con Ollama establecida")
        
        # Generar respuesta
        messages = [
            {"role": "system", "content": "Eres un asistente conciso. Responde en una oración."},
            {"role": "user", "content": "¿Cuál es la capital de España?"}
        ]
        
        print("\n📝 Prompt: ¿Cuál es la capital de España?")
        print("⏳ Generando respuesta...")
        
        response = await model.generate(
            messages,
            temperature=0.7,
            max_tokens=100,
        )
        
        print(f"\n💬 Respuesta: {response.content}")
        print(f"\n📊 Métricas:")
        print(f"   - Modelo: {response.model}")
        print(f"   - Tokens prompt: {response.prompt_tokens}")
        print(f"   - Tokens generados: {response.completion_tokens}")
        print(f"   - Tiempo: {response.generation_time_ms:.0f}ms")
        if response.tokens_per_second:
            print(f"   - Velocidad: {response.tokens_per_second:.1f} tok/s")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def streaming_example():
    """Ejemplo de generación con streaming."""
    print("\n" + "=" * 60)
    print("EJEMPLO 2: Generación con Streaming")
    print("=" * 60)
    
    try:
        model = ModelFactory.create("ollama/llama3.2:1b")
        
        is_healthy = await model.health_check()
        if not is_healthy:
            print("❌ Ollama no está disponible")
            return
        
        messages = [
            {"role": "user", "content": "Cuenta del 1 al 5 en español, un número por línea."}
        ]
        
        print("\n📝 Prompt: Cuenta del 1 al 5")
        print("⏳ Streaming respuesta:\n")
        
        print("💬 ", end="")
        async for chunk in model.generate_stream(messages, max_tokens=50):
            print(chunk, end="", flush=True)
        print("\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def model_manager_example():
    """Ejemplo usando el ModelManager para gestionar múltiples modelos."""
    print("\n" + "=" * 60)
    print("EJEMPLO 3: Usando ModelManager")
    print("=" * 60)
    
    try:
        manager = get_model_manager()
        
        # Precargar modelos
        print("\n⏳ Precargando modelos...")
        results = await manager.preload(
            ["ollama/llama3.2:1b"],
            verify=True,
        )
        
        for model_id, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {model_id}")
        
        # Listar modelos cargados
        loaded = manager.list_loaded()
        print(f"\n📋 Modelos cargados: {loaded}")
        
        # Health check de todos
        print("\n🏥 Health check de todos los modelos:")
        health = await manager.health_check_all()
        for model_id, healthy in health.items():
            status = "✅ Healthy" if healthy else "❌ Unhealthy"
            print(f"   {model_id}: {status}")
        
        # Usar un modelo
        if "ollama/llama3.2:1b" in manager:
            model = await manager.get("ollama/llama3.2:1b")
            response = await model.generate([
                {"role": "user", "content": "Di 'Hola' en tres idiomas"}
            ], max_tokens=50)
            print(f"\n💬 Respuesta: {response.content}")
        
        # Limpiar
        await manager.cleanup()
        print("\n🧹 Modelos descargados")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def model_info_example():
    """Ejemplo de obtención de información del modelo."""
    print("\n" + "=" * 60)
    print("EJEMPLO 4: Información del Modelo")
    print("=" * 60)
    
    try:
        model = ModelFactory.create("ollama/llama3.2:1b")
        
        is_healthy = await model.health_check()
        if not is_healthy:
            print("❌ Ollama no está disponible")
            return
        
        # Obtener información del modelo
        info = await model.get_model_info()
        
        print("\n📄 Información del modelo:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Listar modelos disponibles
        models = await model.list_models()
        print(f"\n📋 Modelos disponibles en Ollama ({len(models)}):")
        for m in models[:5]:  # Mostrar solo los primeros 5
            print(f"   - {m.get('name')}")
        if len(models) > 5:
            print(f"   ... y {len(models) - 5} más")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def embedding_example():
    """Ejemplo de generación de embeddings."""
    print("\n" + "=" * 60)
    print("EJEMPLO 5: Generación de Embeddings")
    print("=" * 60)
    
    try:
        model = ModelFactory.create("ollama/nomic-embed-text")
        
        is_healthy = await model.health_check()
        if not is_healthy:
            print("❌ Modelo de embeddings no disponible")
            print("   Ejecuta: ollama pull nomic-embed-text")
            return
        
        texts = [
            "La inteligencia artificial está transformando la educación",
            "Los robots aprenden como los humanos",
            "Me gusta el helado de chocolate",
        ]
        
        print("\n📝 Textos a embeber:")
        for i, text in enumerate(texts, 1):
            print(f"   {i}. {text}")
        
        print("\n⏳ Generando embeddings...")
        
        response = await model.embed(texts)
        
        print(f"\n📊 Resultados:")
        print(f"   - Dimensiones: {response.dimensions}")
        print(f"   - Número de vectores: {len(response.embeddings)}")
        print(f"   - Tiempo: {response.generation_time_ms:.0f}ms")
        
        # Mostrar primeros valores de cada embedding
        print("\n🔢 Primeros 5 valores de cada embedding:")
        for i, emb in enumerate(response.embeddings, 1):
            preview = ", ".join(f"{v:.4f}" for v in emb[:5])
            print(f"   {i}. [{preview}, ...]")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    """Ejecuta todos los ejemplos."""
    print("🎓 Aula AI Tutor - Ejemplos de Capa de Modelos")
    print("=" * 60)
    
    await basic_generation_example()
    await streaming_example()
    await model_manager_example()
    await model_info_example()
    await embedding_example()
    
    print("\n" + "=" * 60)
    print("✅ Ejemplos completados")
    print("=" * 60)
    
    # Limpiar caché
    ModelFactory.clear_cache()


if __name__ == "__main__":
    asyncio.run(main())
