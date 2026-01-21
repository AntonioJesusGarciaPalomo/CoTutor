"""
Script para probar la API de Ollama directamente sin el adaptador.
"""
import asyncio
import httpx
import time

SIMPLE_SOLVER_PROMPT = """Resuelve el problema. Responde SOLO en JSON:
{"problem_type": "mathematics", "solution": {"steps": [{"step_number": 1, "description": "..."}]}, "final_answer": "...", "hints": [{"level": 1, "content": "..."}]}
"""

async def test_ollama_api():
    print("Testing Ollama API directly with solver prompt...")
    
    payload = {
        "model": "phi:latest",
        "messages": [
            {"role": "system", "content": SIMPLE_SOLVER_PROMPT},
            {"role": "user", "content": "Resuelve: 3x + 7 = 16"}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 500,  # Allow more tokens for JSON
        }
    }
    
    print(f"Payload: {payload}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        start = time.time()
        print("Sending request...")
        
        try:
            response = await client.post(
                "http://localhost:11434/api/chat",
                json=payload
            )
            elapsed = time.time() - start
            print(f"Response received in {elapsed:.2f}s")
            print(f"Status: {response.status_code}")
            print(f"Body: {response.json()}")
        except httpx.TimeoutException:
            print(f"TIMEOUT after {time.time() - start:.2f}s")
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_ollama_api())
