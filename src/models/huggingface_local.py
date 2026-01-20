"""
Adaptador para modelos de HuggingFace Transformers en local.

Este módulo permite cargar y ejecutar modelos directamente desde
HuggingFace Hub usando la librería transformers, sin necesidad
de un servidor intermediario.

Nota: Este adaptador requiere GPU con suficiente VRAM para cargar los modelos.
"""

from __future__ import annotations

import asyncio
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator

from src.core.exceptions import (
    InsufficientVRAMError,
    ModelGenerationError,
    ModelLoadError,
    ModelNotFoundError,
)
from src.core.types import EmbeddingResponse, Message, ModelResponse
from src.models.base import BaseEmbeddingAdapter, BaseModelAdapter


class HuggingFaceLocalAdapter(BaseModelAdapter):
    """
    Adaptador para modelos de HuggingFace Transformers cargados localmente.
    
    Este adaptador carga los modelos directamente en GPU/CPU usando
    la librería transformers, permitiendo un control total sobre
    la configuración de cuantización y memoria.
    
    Attributes:
        repo_id: ID del repositorio en HuggingFace Hub.
        device_map: Estrategia de distribución en dispositivos.
        torch_dtype: Tipo de datos de PyTorch.
        load_in_4bit: Si usar cuantización de 4 bits.
        load_in_8bit: Si usar cuantización de 8 bits.
    
    Example:
        ```python
        adapter = HuggingFaceLocalAdapter(
            model_name="Qwen/Qwen2.5-14B-Instruct",
            load_in_4bit=True,
        )
        
        await adapter.load()
        
        response = await adapter.generate([
            {"role": "user", "content": "Explica qué es un número primo"}
        ])
        ```
    
    Note:
        - Los modelos se cargan de forma lazy (al llamar a load() o en la primera generación)
        - Para liberar memoria, llamar a unload()
        - La cuantización reduce VRAM pero puede afectar la calidad
    """
    
    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        cache_dir: str | None = None,
        trust_remote_code: bool = True,
        attn_implementation: str | None = None,
        max_memory: dict[int | str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Inicializa el adaptador de HuggingFace.
        
        Args:
            model_name: ID del modelo en HuggingFace Hub (ej: "Qwen/Qwen2.5-14B-Instruct").
            device_map: Estrategia de distribución ("auto", "cuda:0", "cpu", etc.).
            torch_dtype: Tipo de datos ("bfloat16", "float16", "float32").
            load_in_4bit: Usar cuantización QLoRA de 4 bits.
            load_in_8bit: Usar cuantización de 8 bits.
            cache_dir: Directorio de caché para modelos descargados.
            trust_remote_code: Permitir código remoto (necesario para algunos modelos).
            attn_implementation: Implementación de atención ("flash_attention_2", "sdpa", etc.).
            max_memory: Límites de memoria por dispositivo.
            **kwargs: Argumentos adicionales para AutoModelForCausalLM.
        """
        super().__init__(
            model_name=model_name,
            backend_name="huggingface",
            **kwargs,
        )
        
        self.repo_id = model_name
        self.device_map = device_map
        self.torch_dtype_str = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.max_memory = max_memory
        self.extra_kwargs = kwargs
        
        self.supports_streaming = True
        self.supports_embeddings = False
        
        # Modelo y tokenizer (cargados de forma lazy)
        self._model = None
        self._tokenizer = None
        self._executor = ThreadPoolExecutor(max_workers=1)
    
    async def load(self) -> None:
        """
        Carga el modelo y tokenizer en memoria.
        
        Esta operación puede tardar varios minutos dependiendo del tamaño
        del modelo y la velocidad de descarga/carga.
        
        Raises:
            ModelLoadError: Si hay un error cargando el modelo.
            InsufficientVRAMError: Si no hay suficiente VRAM.
            ModelNotFoundError: Si el modelo no existe en HuggingFace.
        """
        if self.is_loaded:
            return
        
        # Ejecutar carga en thread pool para no bloquear el event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._load_sync)
    
    def _load_sync(self) -> None:
        """Carga síncrona del modelo (ejecutada en thread pool)."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as e:
            raise ModelLoadError(
                self.repo_id,
                "transformers o torch no están instalados. "
                "Instala con: pip install transformers torch accelerate bitsandbytes",
            ) from e
        
        # Mapear string a dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.torch_dtype_str, torch.bfloat16)
        
        # Configurar cuantización
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        try:
            # Cargar tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.repo_id,
                cache_dir=self.cache_dir,
                trust_remote_code=self.trust_remote_code,
            )
            
            # Configurar pad_token si no existe
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Preparar argumentos del modelo
            model_kwargs: dict[str, Any] = {
                "cache_dir": self.cache_dir,
                "device_map": self.device_map,
                "trust_remote_code": self.trust_remote_code,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["torch_dtype"] = torch_dtype
            
            if self.attn_implementation:
                model_kwargs["attn_implementation"] = self.attn_implementation
            
            if self.max_memory:
                model_kwargs["max_memory"] = self.max_memory
            
            # Añadir kwargs extra
            model_kwargs.update(self.extra_kwargs)
            
            # Cargar modelo
            self._model = AutoModelForCausalLM.from_pretrained(
                self.repo_id,
                **model_kwargs,
            )
            
            self.is_loaded = True
            
        except OSError as e:
            if "404" in str(e) or "not found" in str(e).lower():
                raise ModelNotFoundError(self.repo_id, "huggingface") from e
            raise ModelLoadError(self.repo_id, str(e)) from e
            
        except RuntimeError as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str:
                raise InsufficientVRAMError(
                    self.repo_id,
                    "Desconocido (intenta load_in_4bit=True)",
                ) from e
            raise ModelLoadError(self.repo_id, str(e)) from e
            
        except Exception as e:
            raise ModelLoadError(self.repo_id, str(e)) from e
    
    async def unload(self) -> None:
        """
        Descarga el modelo de memoria.
        
        Libera la VRAM/RAM usada por el modelo.
        """
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        # Forzar limpieza de memoria
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        self.is_loaded = False
    
    async def generate(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Genera una respuesta usando el modelo cargado.
        
        Args:
            messages: Lista de mensajes de la conversación.
            temperature: Temperatura de sampling.
            max_tokens: Máximo de tokens a generar.
            stop: Secuencias de parada (no soportado nativamente, se procesa post-generación).
            top_p: Nucleus sampling.
            top_k: Top-k sampling.
            repetition_penalty: Penalización por repetición.
            do_sample: Si usar sampling (False = greedy).
            **kwargs: Argumentos adicionales para model.generate().
            
        Returns:
            ModelResponse con el texto generado.
        """
        if not self.is_loaded:
            await self.load()
        
        normalized_messages = self._normalize_messages(messages)
        
        # Ejecutar generación en thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor,
            lambda: self._generate_sync(
                normalized_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                **kwargs,
            ),
        )
        
        return result
    
    def _generate_sync(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        do_sample: bool,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generación síncrona (ejecutada en thread pool)."""
        import torch
        
        start_time = time.perf_counter()
        
        # Aplicar template de chat
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback para tokenizers sin chat template
            prompt = self._format_messages_fallback(messages)
        
        # Tokenizar
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # Mover a dispositivo del modelo
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        prompt_tokens = inputs["input_ids"].shape[1]
        
        # Configurar generación
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        
        if do_sample:
            gen_kwargs.update({
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            })
        
        gen_kwargs.update(kwargs)
        
        # Generar
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **gen_kwargs,
            )
        
        # Decodificar solo los tokens nuevos
        new_tokens = outputs[0][prompt_tokens:]
        content = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        completion_tokens = len(new_tokens)
        
        # Procesar secuencias de parada
        if stop:
            for stop_seq in stop:
                if stop_seq in content:
                    content = content.split(stop_seq)[0]
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return self._create_response(
            content=content.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            generation_time_ms=elapsed_ms,
            finish_reason="stop",
        )
    
    def _format_messages_fallback(self, messages: list[dict[str, str]]) -> str:
        """Formato fallback para tokenizers sin chat template."""
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        formatted.append("Assistant:")
        return "\n\n".join(formatted)
    
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
        Genera con streaming usando TextIteratorStreamer.
        
        Note:
            El streaming en HuggingFace requiere threads adicionales.
            Puede ser más lento que la generación completa para respuestas cortas.
        """
        if not self.is_loaded:
            await self.load()
        
        from threading import Thread
        from transformers import TextIteratorStreamer
        
        normalized_messages = self._normalize_messages(messages)
        
        # Preparar prompt
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = self._format_messages_fallback(normalized_messages)
        
        # Tokenizar
        inputs = self._tokenizer(prompt, return_tensors="pt", padding=True)
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Crear streamer
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        
        # Configurar generación
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else 1.0,
            "streamer": streamer,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        
        # Ejecutar generación en thread separado
        thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()
        
        # Iterar sobre el streamer
        for text in streamer:
            if text:
                # Verificar secuencias de parada
                if stop:
                    for stop_seq in stop:
                        if stop_seq in text:
                            text = text.split(stop_seq)[0]
                            yield text
                            return
                yield text
        
        thread.join()
    
    async def health_check(self) -> bool:
        """
        Verifica si el modelo puede cargarse.
        
        Note: No carga el modelo, solo verifica que existe.
        """
        try:
            from huggingface_hub import model_info
            info = model_info(self.repo_id)
            return info is not None
        except Exception:
            return False
    
    async def get_model_info(self) -> dict[str, Any]:
        """Obtiene información del modelo desde HuggingFace Hub."""
        try:
            from huggingface_hub import model_info
            info = model_info(self.repo_id)
            
            return {
                "model": self.repo_id,
                "backend": "huggingface",
                "author": info.author,
                "downloads": info.downloads,
                "likes": info.likes,
                "pipeline_tag": info.pipeline_tag,
                "library_name": info.library_name,
                "tags": info.tags,
                "is_loaded": self.is_loaded,
                "quantization": "4bit" if self.load_in_4bit else ("8bit" if self.load_in_8bit else "none"),
            }
        except Exception as e:
            return {
                "model": self.repo_id,
                "backend": "huggingface",
                "error": str(e),
            }
    
    async def count_tokens(self, text: str) -> int:
        """Cuenta tokens usando el tokenizer real."""
        if not self.is_loaded:
            await self.load()
        
        tokens = self._tokenizer.encode(text)
        return len(tokens)
    
    def __del__(self) -> None:
        """Limpieza al destruir el objeto."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)


class HuggingFaceEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Adaptador para modelos de embedding de HuggingFace.
    
    Usa sentence-transformers para una API más simple de embeddings.
    
    Example:
        ```python
        embedder = HuggingFaceEmbeddingAdapter(
            model_name="BAAI/bge-large-en-v1.5",
        )
        
        response = await embedder.embed(["Texto de ejemplo"])
        print(f"Dimensiones: {response.dimensions}")
        ```
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cuda",
        cache_dir: str | None = None,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            backend_name="huggingface",
            dimensions=dimensions,
            **kwargs,
        )
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1)
    
    async def _load(self) -> None:
        """Carga el modelo de embeddings."""
        if self._model is not None:
            return
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._load_sync)
    
    def _load_sync(self) -> None:
        """Carga síncrona del modelo."""
        try:
            from sentence_transformers import SentenceTransformer
            
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir,
            )
            
            # Obtener dimensiones
            if self.dimensions is None:
                test_emb = self._model.encode(["test"])
                self.dimensions = len(test_emb[0])
                
        except ImportError as e:
            raise ModelLoadError(
                self.model_name,
                "sentence-transformers no está instalado. "
                "Instala con: pip install sentence-transformers",
            ) from e
    
    async def embed(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Genera embeddings para una lista de textos."""
        await self._load()
        
        loop = asyncio.get_event_loop()
        
        start_time = time.perf_counter()
        
        embeddings = await loop.run_in_executor(
            self._executor,
            lambda: self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                **kwargs,
            ).tolist(),
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=self.model_id,
            dimensions=self.dimensions or len(embeddings[0]) if embeddings else 0,
            generation_time_ms=elapsed_ms,
        )
    
    async def health_check(self) -> bool:
        """Verifica si el modelo puede cargarse."""
        try:
            await self._load()
            return True
        except Exception:
            return False
    
    def __del__(self) -> None:
        """Limpieza."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)


__all__ = [
    "HuggingFaceLocalAdapter",
    "HuggingFaceEmbeddingAdapter",
]
