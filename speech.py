import streamlit as st
from dotenv import load_dotenv
import os
import pyttsx3
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define the best-optimized prompt for structured problem-solving
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

        "💡 **Example Solutions:**\n"
        "---\n"
        "🔢 **Example 1:**\n"
        "**Problem:** Find the integral of x.\n\n"
        "**Solution:**\n"
        "1️⃣ **Understanding the Problem:** Compute the indefinite integral of f(x) = x.\n"
        "2️⃣ **Relevant Concepts:** Use the power rule: ∫x^n dx = (x^(n+1))/(n+1) + C.\n"
        "3️⃣ **Step-by-Step Solution:**\n"
        "   - Recognize x as x^1.\n"
        "   - Apply the power rule: increase the exponent by 1, giving x², then divide by the new exponent.\n"
        "   - Add the constant of integration, C.\n"
        "4️⃣ **Final Answer:** (x²)/2 + C.\n"
        "5️⃣ **Verification & Insights:** Differentiating (x²)/2 gives x, confirming correctness.\n\n"
        "---\n"
        "🚗 **Example 2:**\n"
        "**Problem:** A car accelerates from rest at 5 m/s². Find its velocity after 4 seconds.\n\n"
        "**Solution:**\n"
        "1️⃣ **Understanding the Problem:** The car starts from rest and accelerates uniformly.\n"
        "2️⃣ **Relevant Concepts:** Use kinematic equation: v = u + at.\n"
        "3️⃣ **Step-by-Step Solution:**\n"
        "   - Given: u = 0 m/s, a = 5 m/s², t = 4 s.\n"
        "   - Apply formula: v = 0 + (5 × 4) = 20 m/s.\n"
        "4️⃣ **Final Answer:** v = 20 m/s.\n"
        "5️⃣ **Verification & Insights:** The result aligns with expected acceleration; using v² = u² + 2as also gives v = 20 m/s.\n\n"
        "---\n\n"

        "🎯 **Now, solve the following problem using this structured approach:**\n"
        "**Problem:** {problem}\n\n"
        "**Solution:**"
    )
)

# Initialize Groq API model
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="gemma2-9b-it")
llm_chain = LLMChain(llm=llm, prompt=optimized_prompt)

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Streamlit UI Design
st.set_page_config(page_title="STEM Solver 🤖", layout="centered", page_icon="🧠")

st.markdown(
    "<h1 style='text-align: center;'>📚 STEM Problem Solver 🤖</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size:18px;'>Enter a math or physics problem, and I'll solve it step by step! 🚀</p>", 
    unsafe_allow_html=True
)

# User Input
problem = st.text_area("📝 Enter your problem:", placeholder="e.g., What is the integral of x?", height=100)

# Solve Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Solve (Text)"):
        if problem.strip():
            with st.spinner("Thinking... 🤔"):
                response = llm_chain.invoke({"problem": problem})
                solution_text = response['text']
                st.success("✅ Solution Found!")
                st.markdown("### ✨ Solution:")
                st.markdown(f"<div style='background-color:#222831; padding:15px; border-radius:10px; color:white;'>"
                            f"<p style='font-size:16px;'>{solution_text}</p></div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter a valid problem.")

with col2:
    if st.button("🔊 Solve (Speech)"):
        if problem.strip():
            with st.spinner("Speaking... 🎤"):
                response = llm_chain.invoke({"problem": problem})
                solution_text = response['text']
                text_to_speech(solution_text)
                st.success("✅ Solution Read Out!")
        else:
            st.warning("⚠️ Please enter a valid problem.")

# Footer
st.markdown(
    "<br><p style='text-align:center; font-size:14px;'>🚀 Created with ❤️ by an AI-powered tutor! 📖</p>", 
    unsafe_allow_html=True
)
