"""
Sistema de métricas para Aula AI Tutor.

Este módulo proporciona recolección de métricas para monitorear
el rendimiento del sistema, incluyendo:

- Latencia de modelos
- Uso de tokens
- Resultados de guardrails
- Métricas pedagógicas
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean, median, stdev
from threading import Lock
from typing import Any, Generator


@dataclass
class MetricValue:
    """Valor de métrica con timestamp."""
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricStats:
    """Estadísticas agregadas de una métrica."""
    count: int
    sum: float
    min: float
    max: float
    mean: float
    median: float
    stddev: float | None
    p50: float
    p95: float
    p99: float


class MetricsCollector:
    """
    Recolector de métricas en memoria.
    
    Proporciona contadores, gauges, histogramas y timers para
    monitorear el rendimiento del sistema.
    
    Example:
        ```python
        metrics = MetricsCollector()
        
        # Contador
        metrics.increment("requests_total", labels={"model": "qwen2.5"})
        
        # Gauge
        metrics.set_gauge("active_sessions", 5)
        
        # Histograma (latencia)
        metrics.observe("model_latency_ms", 150.5, labels={"model": "qwen2.5"})
        
        # Timer context manager
        with metrics.timer("generation_time"):
            await model.generate(...)
        
        # Obtener estadísticas
        stats = metrics.get_stats("model_latency_ms")
        print(f"P95 latencia: {stats.p95}ms")
        ```
    """
    
    def __init__(self, max_samples: int = 10000) -> None:
        """
        Inicializa el recolector de métricas.
        
        Args:
            max_samples: Máximo de muestras a mantener por métrica.
        """
        self.max_samples = max_samples
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[MetricValue]] = defaultdict(list)
        self._lock = Lock()
    
    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Incrementa un contador.
        
        Args:
            name: Nombre del contador.
            value: Valor a incrementar (default: 1).
            labels: Etiquetas opcionales.
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += value
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Establece el valor de un gauge.
        
        Args:
            name: Nombre del gauge.
            value: Valor a establecer.
            labels: Etiquetas opcionales.
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value
    
    def observe(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Registra una observación en un histograma.
        
        Args:
            name: Nombre del histograma.
            value: Valor observado.
            labels: Etiquetas opcionales.
        """
        key = self._make_key(name, labels)
        metric = MetricValue(value=value, labels=labels or {})
        
        with self._lock:
            self._histograms[key].append(metric)
            # Limitar tamaño
            if len(self._histograms[key]) > self.max_samples:
                self._histograms[key] = self._histograms[key][-self.max_samples:]
    
    @contextmanager
    def timer(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> Generator[None, None, None]:
        """
        Context manager para medir tiempo de ejecución.
        
        Args:
            name: Nombre de la métrica de tiempo.
            labels: Etiquetas opcionales.
            
        Yields:
            None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.observe(name, elapsed_ms, labels)
    
    def get_counter(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> float:
        """Obtiene el valor de un contador."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0.0)
    
    def get_gauge(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> float | None:
        """Obtiene el valor de un gauge."""
        key = self._make_key(name, labels)
        return self._gauges.get(key)
    
    def get_stats(
        self,
        name: str,
        labels: dict[str, str] | None = None,
        since: datetime | None = None,
    ) -> MetricStats | None:
        """
        Obtiene estadísticas agregadas de un histograma.
        
        Args:
            name: Nombre del histograma.
            labels: Etiquetas opcionales para filtrar.
            since: Solo incluir valores desde esta fecha.
            
        Returns:
            MetricStats con estadísticas o None si no hay datos.
        """
        key = self._make_key(name, labels)
        
        with self._lock:
            values_list = self._histograms.get(key, [])
            
            if since:
                values_list = [v for v in values_list if v.timestamp >= since]
            
            if not values_list:
                return None
            
            values = [v.value for v in values_list]
            sorted_values = sorted(values)
            n = len(values)
            
            return MetricStats(
                count=n,
                sum=sum(values),
                min=min(values),
                max=max(values),
                mean=mean(values),
                median=median(values),
                stddev=stdev(values) if n > 1 else None,
                p50=self._percentile(sorted_values, 50),
                p95=self._percentile(sorted_values, 95),
                p99=self._percentile(sorted_values, 99),
            )
    
    def get_all_metrics(self) -> dict[str, Any]:
        """
        Obtiene todas las métricas como diccionario.
        
        Returns:
            Diccionario con todos los contadores, gauges e histogramas.
        """
        with self._lock:
            result: dict[str, Any] = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {},
            }
            
            for key, values in self._histograms.items():
                if values:
                    sorted_vals = sorted(v.value for v in values)
                    result["histograms"][key] = {
                        "count": len(values),
                        "min": min(sorted_vals),
                        "max": max(sorted_vals),
                        "mean": mean(sorted_vals),
                        "p95": self._percentile(sorted_vals, 95),
                    }
            
            return result
    
    def reset(self) -> None:
        """Reinicia todas las métricas."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
    
    def _make_key(
        self,
        name: str,
        labels: dict[str, str] | None,
    ) -> str:
        """Genera una clave única para la métrica."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    @staticmethod
    def _percentile(sorted_values: list[float], p: float) -> float:
        """Calcula el percentil p de una lista ordenada."""
        if not sorted_values:
            return 0.0
        
        n = len(sorted_values)
        k = (n - 1) * (p / 100)
        f = int(k)
        c = f + 1
        
        if c >= n:
            return sorted_values[-1]
        
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


# Métricas específicas para el sistema educativo
class PedagogicalMetrics:
    """
    Métricas específicas para evaluación pedagógica.
    
    Rastrea métricas relacionadas con la efectividad de la tutoría.
    """
    
    def __init__(self, collector: MetricsCollector | None = None) -> None:
        self.collector = collector or MetricsCollector()
        self._sessions: dict[str, dict[str, Any]] = {}
    
    def start_session(self, session_id: str, problem_type: str) -> None:
        """Inicia una sesión de tutoría."""
        self._sessions[session_id] = {
            "start_time": datetime.now(),
            "problem_type": problem_type,
            "questions_by_tutor": 0,
            "responses_by_tutor": 0,
            "hints_given": [],
            "student_attempts": 0,
        }
        self.collector.increment("sessions_started", labels={"type": problem_type})
    
    def record_tutor_message(
        self,
        session_id: str,
        contains_question: bool,
        hint_level: int | None = None,
    ) -> None:
        """Registra un mensaje del tutor."""
        if session_id not in self._sessions:
            return
        
        session = self._sessions[session_id]
        session["responses_by_tutor"] += 1
        
        if contains_question:
            session["questions_by_tutor"] += 1
        
        if hint_level is not None:
            session["hints_given"].append(hint_level)
            self.collector.increment(
                "hints_given",
                labels={"level": str(hint_level)},
            )
    
    def record_student_attempt(self, session_id: str, is_correct: bool) -> None:
        """Registra un intento del estudiante."""
        if session_id not in self._sessions:
            return
        
        self._sessions[session_id]["student_attempts"] += 1
        
        label = "correct" if is_correct else "incorrect"
        self.collector.increment("student_attempts", labels={"result": label})
    
    def end_session(
        self,
        session_id: str,
        student_solved: bool,
    ) -> dict[str, Any] | None:
        """
        Finaliza una sesión y calcula métricas.
        
        Returns:
            Diccionario con métricas de la sesión.
        """
        if session_id not in self._sessions:
            return None
        
        session = self._sessions.pop(session_id)
        
        duration = (datetime.now() - session["start_time"]).total_seconds()
        
        # Calcular ratio de preguntas
        total_messages = session["responses_by_tutor"]
        question_ratio = (
            session["questions_by_tutor"] / total_messages
            if total_messages > 0
            else 0
        )
        
        # Registrar métricas
        self.collector.observe("session_duration_seconds", duration)
        self.collector.observe("question_ratio", question_ratio)
        self.collector.observe("attempts_to_solution", session["student_attempts"])
        
        result_label = "solved" if student_solved else "abandoned"
        self.collector.increment("sessions_completed", labels={"result": result_label})
        
        return {
            "session_id": session_id,
            "duration_seconds": duration,
            "problem_type": session["problem_type"],
            "question_ratio": question_ratio,
            "hints_given": session["hints_given"],
            "student_attempts": session["student_attempts"],
            "student_solved": student_solved,
        }
    
    def get_summary(self) -> dict[str, Any]:
        """Obtiene resumen de métricas pedagógicas."""
        return {
            "active_sessions": len(self._sessions),
            "session_duration": self.collector.get_stats("session_duration_seconds"),
            "question_ratio": self.collector.get_stats("question_ratio"),
            "attempts_to_solution": self.collector.get_stats("attempts_to_solution"),
            "hints_by_level": {
                str(i): self.collector.get_counter("hints_given", {"level": str(i)})
                for i in range(1, 4)
            },
        }


# Singleton global de métricas
_metrics_collector: MetricsCollector | None = None
_pedagogical_metrics: PedagogicalMetrics | None = None


def get_metrics() -> MetricsCollector:
    """Obtiene el recolector de métricas global."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_pedagogical_metrics() -> PedagogicalMetrics:
    """Obtiene las métricas pedagógicas globales."""
    global _pedagogical_metrics
    if _pedagogical_metrics is None:
        _pedagogical_metrics = PedagogicalMetrics(get_metrics())
    return _pedagogical_metrics


__all__ = [
    "MetricsCollector",
    "MetricValue",
    "MetricStats",
    "PedagogicalMetrics",
    "get_metrics",
    "get_pedagogical_metrics",
]
