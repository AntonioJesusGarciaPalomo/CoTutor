"""
Configuración centralizada del sistema Aula AI Tutor.

Este módulo proporciona una configuración tipada y validada usando Pydantic Settings.
Soporta carga desde variables de entorno y archivos .env.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Enumeraciones
# =============================================================================

class ModelBackend(str, Enum):
    """Backends de modelos soportados."""
    OLLAMA = "ollama"
    OPENAI_LOCAL = "openai_local"
    HUGGINGFACE = "huggingface"


class AgentRole(str, Enum):
    """Roles de agentes en el sistema."""
    SOLVER = "solver"
    TUTOR = "tutor"


class LogLevel(str, Enum):
    """Niveles de logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# =============================================================================
# Modelos de Configuración
# =============================================================================

class OllamaConfig(BaseModel):
    """Configuración para backend Ollama."""
    base_url: str = "http://localhost:11434"
    timeout: int = 120
    
    model_config = {"extra": "allow"}


class OpenAILocalConfig(BaseModel):
    """Configuración para backends compatibles con OpenAI API."""
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "not-needed"
    timeout: int = 120
    
    model_config = {"extra": "allow"}


class HuggingFaceConfig(BaseModel):
    """Configuración para modelos de HuggingFace."""
    cache_dir: str = "~/.cache/huggingface"
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    
    model_config = {"extra": "allow"}
    
    @field_validator("cache_dir")
    @classmethod
    def expand_cache_dir(cls, v: str) -> str:
        """Expande ~ en la ruta del caché."""
        return str(Path(v).expanduser())


class ModelDefaults(BaseModel):
    """Valores por defecto para modelos."""
    solver_model: str = "ollama/qwen2.5:14b"
    tutor_model: str = "ollama/llama3.1:8b"
    embedding_model: str = "ollama/nomic-embed-text"
    solver_temperature: float = 0.1
    tutor_temperature: float = 0.7
    solver_max_tokens: int = 4096
    tutor_max_tokens: int = 1024


class GuardrailsConfig(BaseModel):
    """Configuración para el sistema de guardrails."""
    # Detector de manipulación
    manipulation_detection_enabled: bool = True
    manipulation_threshold: float = 0.8
    
    # Detector de fuga de solución
    solution_leak_detection_enabled: bool = True
    semantic_similarity_threshold: float = 0.75
    key_answer_match_threshold: float = 0.0
    step_revelation_threshold: float = 0.3
    
    # Validador pedagógico
    pedagogical_validation_enabled: bool = True
    min_question_ratio: float = 0.3  # Al menos 30% de respuestas deben ser preguntas
    
    # Modelo de embeddings para detección
    embedding_model: str = "ollama/nomic-embed-text"


class A2AConfig(BaseModel):
    """Configuración para el protocolo A2A."""
    solver_host: str = "localhost"
    solver_port: int = 8001
    tutor_host: str = "localhost"
    tutor_port: int = 8002
    
    # Timeouts
    connection_timeout: int = 30
    request_timeout: int = 120
    
    # Reintentos
    max_retries: int = 3
    retry_delay: float = 1.0


class PerformanceConfig(BaseModel):
    """Configuración de rendimiento."""
    target_tokens_per_second: int = 30
    generation_timeout: int = 120
    embedding_batch_size: int = 32
    
    # Caché del Solver
    solver_cache_enabled: bool = True
    solver_cache_max_size: int = 1000
    solver_cache_ttl: int = 3600


class LoggingConfig(BaseModel):
    """Configuración de logging."""
    level: LogLevel = LogLevel.INFO
    format: str = "json"
    include_timestamps: bool = True
    log_model_inputs: bool = False
    log_model_outputs: bool = False
    log_file: str | None = None


# =============================================================================
# Settings Principal
# =============================================================================

class Settings(BaseSettings):
    """
    Configuración principal del sistema Aula AI Tutor.
    
    Los valores pueden ser sobrescritos mediante variables de entorno
    con el prefijo AULA_, por ejemplo:
    - AULA_DEBUG=true
    - AULA_OLLAMA_BASE_URL=http://192.168.1.100:11434
    """
    
    model_config = SettingsConfigDict(
        env_prefix="AULA_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Modo debug
    debug: bool = False
    
    # Rutas del proyecto
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    config_dir: Path = Field(default_factory=lambda: Path(__file__).parent)
    
    # Archivo de configuración de modelos
    models_config_file: str = "models.yaml"
    
    # Configuraciones de subsistemas
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    openai_local: OpenAILocalConfig = Field(default_factory=OpenAILocalConfig)
    huggingface: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)
    model_defaults: ModelDefaults = Field(default_factory=ModelDefaults)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    a2a: A2AConfig = Field(default_factory=A2AConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Cache de configuración de modelos YAML
    _models_config_cache: dict[str, Any] | None = None
    
    def get_models_config(self) -> dict[str, Any]:
        """
        Carga y cachea la configuración de modelos desde YAML.
        
        Returns:
            Diccionario con la configuración de modelos.
        """
        if self._models_config_cache is None:
            config_path = self.config_dir / self.models_config_file
            if config_path.exists():
                with open(config_path) as f:
                    self._models_config_cache = yaml.safe_load(f)
            else:
                self._models_config_cache = {}
        return self._models_config_cache
    
    def get_model_info(self, model_id: str) -> dict[str, Any] | None:
        """
        Obtiene información de un modelo específico.
        
        Args:
            model_id: Identificador del modelo (ej: "ollama/qwen2.5:14b")
            
        Returns:
            Diccionario con información del modelo o None si no existe.
        """
        models_config = self.get_models_config()
        
        # Parsear el model_id: backend/model_name
        if "/" not in model_id:
            return None
            
        backend, model_name = model_id.split("/", 1)
        
        if backend not in models_config:
            return None
            
        backend_config = models_config[backend]
        models = backend_config.get("models", {})
        
        # Buscar el modelo (puede estar con : o sin)
        for key, info in models.items():
            if key == model_name or info.get("name") == model_name:
                return {
                    "backend": backend,
                    "model_name": model_name,
                    **info
                }
        
        return None
    
    def get_backend_config(self, backend: ModelBackend | str) -> dict[str, Any]:
        """
        Obtiene la configuración de un backend específico.
        
        Args:
            backend: Tipo de backend (ollama, openai_local, huggingface)
            
        Returns:
            Diccionario con la configuración del backend.
        """
        if isinstance(backend, ModelBackend):
            backend = backend.value
            
        if backend == "ollama":
            return self.ollama.model_dump()
        elif backend == "openai_local":
            return self.openai_local.model_dump()
        elif backend == "huggingface":
            return self.huggingface.model_dump()
        else:
            raise ValueError(f"Backend desconocido: {backend}")


@lru_cache
def get_settings() -> Settings:
    """
    Obtiene la instancia singleton de Settings.
    
    Esta función usa caché para evitar recargar la configuración
    múltiples veces.
    
    Returns:
        Instancia de Settings configurada.
    """
    return Settings()


# =============================================================================
# Funciones de utilidad
# =============================================================================

def parse_model_id(model_id: str) -> tuple[str, str]:
    """
    Parsea un identificador de modelo en backend y nombre.
    
    Args:
        model_id: Identificador del modelo (ej: "ollama/qwen2.5:14b")
        
    Returns:
        Tupla (backend, model_name)
        
    Raises:
        ValueError: Si el formato es inválido.
    """
    if "/" not in model_id:
        raise ValueError(
            f"Formato de model_id inválido: {model_id}. "
            f"Usa el formato 'backend/model_name' (ej: 'ollama/qwen2.5:14b')"
        )
    
    parts = model_id.split("/", 1)
    return parts[0], parts[1]


def get_default_model(role: AgentRole) -> str:
    """
    Obtiene el modelo por defecto para un rol.
    
    Args:
        role: Rol del agente (solver o tutor)
        
    Returns:
        Identificador del modelo por defecto.
    """
    settings = get_settings()
    
    if role == AgentRole.SOLVER:
        return settings.model_defaults.solver_model
    elif role == AgentRole.TUTOR:
        return settings.model_defaults.tutor_model
    else:
        raise ValueError(f"Rol desconocido: {role}")


# Crear archivo __init__.py para el módulo config
__all__ = [
    "Settings",
    "get_settings",
    "ModelBackend",
    "AgentRole",
    "LogLevel",
    "OllamaConfig",
    "OpenAILocalConfig",
    "HuggingFaceConfig",
    "ModelDefaults",
    "GuardrailsConfig",
    "A2AConfig",
    "PerformanceConfig",
    "LoggingConfig",
    "parse_model_id",
    "get_default_model",
]
