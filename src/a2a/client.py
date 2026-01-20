"""
Cliente para el protocolo A2A.

Permite la comunicación entre agentes y clientes externos con el servicio.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import get_settings
from src.core.exceptions import ConnectionError
from src.core.types import (
    StructuredSolution,
    StudentInput,
    TutorResponse,
)


class A2AClient:
    """
    Cliente para interactuar con los servicios de Solver y Tutor.
    """
    
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Inicializa el cliente A2A.
        
        Args:
            base_url: URL base del servicio (opcional).
            timeout: Timeout para peticiones.
        """
        settings = get_settings()
        
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            host = settings.a2a.solver_host
            port = settings.a2a.solver_port
            self.base_url = f"http://{host}:{port}"
            
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        
    async def _get_client(self) -> httpx.AsyncClient:
        """Obtiene el cliente HTTP."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client
    
    async def close(self) -> None:
        """Cierra el cliente."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self) -> "A2AClient":
        await self._get_client()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
    
    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def solve_problem(
        self,
        problem_text: str,
        domain_hint: str | None = None,
    ) -> StructuredSolution:
        """
        Solicita al Solver resolver un problema.
        
        Args:
            problem_text: Texto del problema.
            domain_hint: Dominio opcional.
            
        Returns:
            Solución estructurada.
        """
        client = await self._get_client()
        
        payload = {
            "problem_text": problem_text,
            "domain_hint": domain_hint,
        }
        
        try:
            response = await client.post("/solver/solve", json=payload)
            response.raise_for_status()
            
            return StructuredSolution(**response.json())
            
        except httpx.ConnectError as e:
            raise ConnectionError(f"No se pudo conectar al servicio Solver: {e}")
        except httpx.HTTPStatusError as e:
            raise ConnectionError(f"Error en servicio Solver: {e.response.text}")

    async def start_tutoring_session(
        self,
        problem_text: str,
        solution: StructuredSolution,
    ) -> UUID:
        """
        Inicia una sesión de tutoría.
        
        Args:
            problem_text: Texto del problema.
            solution: Solución estructurada.
            
        Returns:
            ID de la sesión.
        """
        client = await self._get_client()
        
        payload = {
            "problem_text": problem_text,
            "solution": solution.model_dump(mode="json"),
        }
        
        try:
            response = await client.post("/tutor/session", json=payload)
            response.raise_for_status()
            
            data = response.json()
            return UUID(data["session_id"])
            
        except httpx.ConnectError as e:
            raise ConnectionError(f"No se pudo conectar al servicio Tutor: {e}")
            
    async def send_message(
        self,
        session_id: UUID,
        message: str,
    ) -> TutorResponse:
        """
        Envía un mensaje al Tutor.
        
        Args:
            session_id: ID de la sesión.
            message: Mensaje del estudiante.
            
        Returns:
            Respuesta del tutor.
        """
        client = await self._get_client()
        
        payload = {
            "session_id": str(session_id),
            "message": message,
        }
        
        try:
            response = await client.post("/tutor/chat", json=payload)
            response.raise_for_status()
            
            return TutorResponse(**response.json())
            
        except httpx.ConnectError as e:
            raise ConnectionError(f"No se pudo conectar al servicio Tutor: {e}")
