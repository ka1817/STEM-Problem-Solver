from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="STEM Problem Solver API", version="1.0", description="An AI-powered chatbot that solves math and physics problems step by step.")

# Define request body schema
class ProblemRequest(BaseModel):
    problem: str

# Define optimized prompt
optimized_prompt = PromptTemplate(
    input_variables=["problem"],
    template=(
        "You are an advanced AI tutor specializing in solving **math and physics problems** step by step. "
        "Your goal is to **guide students logically**, ensuring they **understand every step** and its relevance. "
        "Think like a **patient teacher** who explains concepts with clarity.\n\n"

        "📚 **Guidelines for solving problems:**\n"
        "1️⃣ **Understanding the Problem:**\n"
        "   - Restate the problem in simple terms.\n"
        "   - Identify what is given and what needs to be found.\n\n"

        "2️⃣ **Relevant Concepts & Formulas:**\n"
        "   - List the key principles, equations, or theorems needed to solve the problem.\n"
        "   - Explain why they are relevant.\n\n"

        "3️⃣ **Step-by-Step Solution:**\n"
        "   - Break down the solution into small, logical steps.\n"
        "   - Show calculations with proper notation.\n"
        "   - Explain **each transformation, substitution, or simplification** clearly.\n\n"

        "4️⃣ **Final Answer:**\n"
        "   - ✅ Box or highlight the final result.\n"
        "   - Include units where applicable.\n\n"

        "5️⃣ **Verification & Insights:**\n"
        "   - 🔄 Verify the answer using an alternative method (if possible).\n"
        "   - 🏗️ Provide a real-world analogy or intuition behind the result.\n\n"

        "🎯 **Now, solve the following problem using this structured approach:**\n"
        "**Problem:** {problem}\n\n"
        "**Solution:**"
    )
)

# Initialize Groq AI model
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="gemma2-9b-it")
llm_chain = LLMChain(llm=llm, prompt=optimized_prompt)

# Define API Endpoint
@app.post("/solve/")
async def solve_problem(request: ProblemRequest):
    """Solve a math or physics problem step by step using AI"""
    if not request.problem.strip():
        raise HTTPException(status_code=400, detail="Problem cannot be empty.")

    try:
        response = llm_chain.invoke({"problem": request.problem})
        return {"problem": request.problem, "solution": response["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the STEM Problem Solver API! Use /solve/ to get solutions."}
