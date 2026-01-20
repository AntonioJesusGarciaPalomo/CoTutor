"""
Módulo de utilidades para Aula AI Tutor.

Incluye:
- Sistema de logging estructurado
- Recolección de métricas
"""

from src.utils.logging import (
    LogContext,
    configure_logging,
    get_logger,
    log_a2a_message,
    log_guardrail_result,
    log_model_call,
)
from src.utils.metrics import (
    MetricStats,
    MetricValue,
    MetricsCollector,
    PedagogicalMetrics,
    get_metrics,
    get_pedagogical_metrics,
)

__all__ = [
    # Logging
    "configure_logging",
    "get_logger",
    "LogContext",
    "log_model_call",
    "log_guardrail_result",
    "log_a2a_message",
    # Metrics
    "MetricsCollector",
    "MetricValue",
    "MetricStats",
    "PedagogicalMetrics",
    "get_metrics",
    "get_pedagogical_metrics",
]
