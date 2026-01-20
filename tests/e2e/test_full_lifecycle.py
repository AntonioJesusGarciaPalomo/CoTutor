"""
Tests End-to-End (E2E) completos para el ciclo de vida de tutoría.
"""

import pytest
from fastapi.testclient import TestClient

from src.services.app import app
from src.core.types import ProblemType, DifficultyLevel

client = TestClient(app)

@pytest.mark.e2e
class TestFullLifecycle:
    """Tests del ciclo de vida completo Solver -> Tutor."""
    
    def test_solve_and_tutor_flow(self, mocker):
        """
        Simula un flujo completo:
        1. Resolver un problema.
        2. Iniciar sesión de tutoría con la solución.
        3. Interactuar con el tutor.
        """
        import os
        is_real_mode = os.environ.get("E2E_REAL_MODELS") == "true"
        
        # ---------------------------------------------------------------------
        # 1. Configurar Mocks (SOLO SI NO ES MODO REAL)
        # ---------------------------------------------------------------------
        if not is_real_mode:
            # Solver
            mock_solve = mocker.patch("src.agents.solver.SolverAgent.solve")
            
            from src.core.types import StructuredSolution, ProblemType, DifficultyLevel
            from uuid import uuid4
            
            fake_solution = StructuredSolution(
                problem_id=uuid4(),
                problem_text="Resuelve 2x = 8",
                problem_type=ProblemType.MATHEMATICS,
                difficulty=DifficultyLevel.BASIC,
                steps=[],
                hints=[],
                final_answer="x = 4",
                verification="2(4)=8",
                solver_model="test-model"
            )
            mock_solve.return_value = fake_solution
            
            # Tutor
            mock_start = mocker.patch("src.agents.tutor.TutorAgent.start_session")
            mock_session_obj = mocker.Mock()
            mock_session_obj.session_id = "12345678-1234-5678-1234-567812345678"
            mock_start.return_value = mock_session_obj
            
            mock_respond = mocker.patch("src.agents.tutor.TutorAgent.respond")
            from src.core.types import TutorResponse
            mock_respond.return_value = TutorResponse(
                content="¡Hola! Vamos a empezar.",
                tutor_model="test-tutor"
            )
        
        # ---------------------------------------------------------------------
        # 2. Ejecutar flujo E2E contra la API
        # ---------------------------------------------------------------------
        
        # A. Solicitar solución
        problem_text = "Resuelve 2x = 8"
        print(f"   [Step 1] Requesting solution for: {problem_text}")
        solve_response = client.post("/solver/solve", json={
            "problem_text": problem_text
        })
        
        if solve_response.status_code != 200:
            print(f"Error Solver: {solve_response.text}")
            
        assert solve_response.status_code == 200
        solution_data = solve_response.json()
        
        if not is_real_mode:
            assert solution_data["final_answer"] == "x = 4"
        else:
            # En modo real, solo verificamos que haya una respuesta coherente
            assert "final_answer" in solution_data
            assert len(solution_data["final_answer"]) > 0
            print(f"   [Step 1] Solution received: {solution_data['final_answer']}")
        
        # B. Iniciar sesión de tutoría
        print("   [Step 2] Starting tutoring session...")
        session_response = client.post("/tutor/session", json={
            "problem_text": problem_text,
            "solution": solution_data
        })
        assert session_response.status_code == 200
        session_id = session_response.json()["session_id"]
        
        if not is_real_mode:
            assert session_id == "12345678-1234-5678-1234-567812345678"
        
        # C. Chat con el tutor
        print(f"   [Step 3] Chatting with tutor (Session: {session_id})")
        chat_response = client.post("/tutor/chat", json={
            "session_id": session_id,
            "message": "Hola, ¿por dónde empiezo?"
        })
        assert chat_response.status_code == 200
        content = chat_response.json()["content"]
        
        if not is_real_mode:
            assert content == "¡Hola! Vamos a empezar."
        else:
            assert len(content) > 0
            print(f"   [Step 3] Tutor response: {content}")

