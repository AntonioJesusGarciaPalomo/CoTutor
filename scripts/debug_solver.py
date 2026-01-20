
import asyncio
import traceback
from src.agents.solver import SolverAgent
from config.settings import get_settings

async def main():
    settings = get_settings()
    model_id = settings.model_defaults.solver_model
    print(f"Testing Solver instantiation with model: {model_id}")
    
    try:
        print("1. Creating SolverAgent...")
        agent = await SolverAgent.create(model_id)
        print("   Success!")
        
        print("2. Attempting to solve simple problem...")
        problem = "2+2=4"
        solution = await agent.solve(problem)
        print("   Success!")
        print(f"   Answer: {solution.final_answer}")
        
    except Exception:
        print("\n!!! ERROR ENCOUNTERED !!!")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
