"""
Sistema de logging estructurado para Aula AI Tutor.

Este módulo configura logging usando structlog para proporcionar
logs estructurados en formato JSON o consola legible.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Any

import structlog
from structlog.types import Processor

from config.settings import LogLevel, get_settings


def configure_logging(
    level: LogLevel | str = LogLevel.INFO,
    format: str = "json",
    include_timestamps: bool = True,
    log_file: str | None = None,
) -> None:
    """
    Configura el sistema de logging.
    
    Args:
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format: Formato de salida ("json" o "console").
        include_timestamps: Si incluir timestamps en los logs.
        log_file: Ruta opcional a archivo de log.
    """
    if isinstance(level, str):
        level = LogLevel(level.upper())
    
    # Configurar nivel de logging estándar
    logging.basicConfig(
        level=getattr(logging, level.value),
        format="%(message)s",
        stream=sys.stdout,
    )
    
    # Procesadores base
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if include_timestamps:
        processors.insert(0, structlog.processors.TimeStamper(fmt="iso"))
    
    # Procesador final según formato
    if format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        ))
    
    # Configurar structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.value)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configurar archivo de log si se especifica
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.value))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Obtiene un logger configurado.
    
    Args:
        name: Nombre del logger (opcional).
        
    Returns:
        Logger estructurado listo para usar.
    
    Example:
        ```python
        logger = get_logger(__name__)
        logger.info("Mensaje", extra_field="valor")
        ```
    """
    logger = structlog.get_logger(name)
    return logger


class LogContext:
    """
    Context manager para añadir contexto temporal a los logs.
    
    Example:
        ```python
        with LogContext(request_id="abc123", user_id="user1"):
            logger.info("Procesando solicitud")
            # Todos los logs dentro tendrán request_id y user_id
        ```
    """
    
    def __init__(self, **kwargs: Any) -> None:
        self.context = kwargs
        self._token = None
    
    def __enter__(self) -> "LogContext":
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self
    
    def __exit__(self, *args: Any) -> None:
        structlog.contextvars.unbind_contextvars(*self.context.keys())


def log_model_call(
    logger: structlog.BoundLogger,
    model_id: str,
    operation: str,
    **kwargs: Any,
) -> None:
    """
    Log estandarizado para llamadas a modelos.
    
    Args:
        logger: Logger a usar.
        model_id: Identificador del modelo.
        operation: Tipo de operación (generate, embed, etc.).
        **kwargs: Datos adicionales.
    """
    logger.info(
        "model_call",
        model_id=model_id,
        operation=operation,
        **kwargs,
    )


def log_guardrail_result(
    logger: structlog.BoundLogger,
    guardrail_name: str,
    result: str,
    score: float | None = None,
    **kwargs: Any,
) -> None:
    """
    Log estandarizado para resultados de guardrails.
    
    Args:
        logger: Logger a usar.
        guardrail_name: Nombre del guardrail.
        result: Resultado (pass, warn, block).
        score: Puntuación opcional.
        **kwargs: Datos adicionales.
    """
    log_level = "warning" if result == "block" else "info"
    getattr(logger, log_level)(
        "guardrail_result",
        guardrail=guardrail_name,
        result=result,
        score=score,
        **kwargs,
    )


def log_a2a_message(
    logger: structlog.BoundLogger,
    direction: str,
    message_type: str,
    sender: str,
    recipient: str,
    **kwargs: Any,
) -> None:
    """
    Log estandarizado para mensajes A2A.
    
    Args:
        logger: Logger a usar.
        direction: "sent" o "received".
        message_type: Tipo de mensaje A2A.
        sender: Agente emisor.
        recipient: Agente receptor.
        **kwargs: Datos adicionales.
    """
    logger.info(
        "a2a_message",
        direction=direction,
        message_type=message_type,
        sender=sender,
        recipient=recipient,
        **kwargs,
    )


# Configurar logging con settings por defecto al importar
def _init_logging() -> None:
    """Inicializa logging con configuración del sistema."""
    try:
        settings = get_settings()
        configure_logging(
            level=settings.logging.level,
            format=settings.logging.format,
            include_timestamps=settings.logging.include_timestamps,
            log_file=settings.logging.log_file,
        )
    except Exception:
        # Fallback a configuración básica si hay error
        configure_logging()


# Auto-inicializar
_init_logging()


__all__ = [
    "configure_logging",
    "get_logger",
    "LogContext",
    "log_model_call",
    "log_guardrail_result",
    "log_a2a_message",
]
