"""
Tests de integración para la API A2A.
"""

import pytest
from fastapi.testclient import TestClient

from src.services.app import app
from src.core.types import StructuredSolution, SolutionStep, Hint, HintLevel, ProblemType, DifficultyLevel

client = TestClient(app)

@pytest.fixture
def mock_solution():
    return StructuredSolution(
        problem_text="2x + 3 = 7",
        problem_type=ProblemType.MATHEMATICS,
        difficulty=DifficultyLevel.BASIC,
        steps=[
            SolutionStep(
                step_number=1,
                description="Resta 3",
                reasoning="Aislar x",
                result="2x = 4"
            )
        ],
        final_answer="x = 2",
        hints=[
            Hint(level=HintLevel.SUBTLE, content="Pista 1"),
            Hint(level=HintLevel.MODERATE, content="Pista 2"),
            Hint(level=HintLevel.DIRECT, content="Pista 3"),
        ]
    )

def test_health_check():
    """Verifica el endpoint de salud."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_solve_endpoint_mock(mocker):
    """Test del endpoint /solver/solve con mock."""
    mock_solve = mocker.patch("src.agents.solver.SolverAgent.solve")
    mock_solve.return_value = StructuredSolution(
        problem_text="test",
        problem_type=ProblemType.GENERAL,
        difficulty=DifficultyLevel.BASIC,
        steps=[],
        final_answer="42",
        hints=[]
    )
    
    response = client.post("/solver/solve", json={
        "problem_text": "Meaning of life"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["final_answer"] == "42"

@pytest.mark.asyncio
async def test_tutor_session_flow(mocker, mock_solution):
    """Test del flujo de sesión de tutoría."""
    # Mockear dependencias del Tutor
    mocker.patch("src.agents.tutor.SessionManager.create_session")
    # ... esto es complejo de mockear completamente si no usamos inyección de dependencias más estricta
    # Para este nivel de integración, verificamos que el endpoint responda
    
    # Asumimos que el TutorAgent usa mocks internos o lo mockeamos
    mock_start = mocker.patch("src.agents.tutor.TutorAgent.start_session")
    mock_session = mocker.Mock()
    mock_session.session_id = "123e4567-e89b-12d3-a456-426614174000"
    mock_start.return_value = mock_session
    
    response = client.post("/tutor/session", json={
        "problem_text": "2x + 3 = 7",
        "solution": mock_solution.model_dump(mode="json")
    })
    
    assert response.status_code == 200
    assert response.json()["session_id"] == "123e4567-e89b-12d3-a456-426614174000"

