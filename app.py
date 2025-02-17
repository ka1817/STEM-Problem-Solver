import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
import base64
from gtts import gTTS  # ✅ Use gTTS instead of pyttsx3
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ✅ Exact optimized prompt as provided
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

# Initialize Groq API model
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="gemma2-9b-it")
llm_chain = LLMChain(llm=llm, prompt=optimized_prompt)

# Function to generate speech using gTTS and return Base64-encoded audio
def text_to_speech(text):
    """Generate gTTS audio and return Base64 encoded string."""
    tts = gTTS(text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts.save(temp_audio.name)
        with open(temp_audio.name, "rb") as audio_file:
            audio_bytes = audio_file.read()
            encoded_audio = base64.b64encode(audio_bytes).decode()
        os.remove(temp_audio.name)
    return encoded_audio

# Streamlit UI Design
st.set_page_config(page_title="STEM Solver 🤖", layout="centered", page_icon="🧠")

st.markdown("<h1 style='text-align: center;'>📚 STEM Problem Solver 🤖</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Enter a math or physics problem, and I'll solve it step by step! 🚀</p>", unsafe_allow_html=True)

# User Input
problem = st.text_area("📝 Enter your problem:", placeholder="e.g., What is the integral of x?", height=100)

# Solve (Text) Button
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

# Solve (Speech) Button
if st.button("🔊 Solve (Speech)"):
    if problem.strip():
        with st.spinner("Speaking... 🎤"):
            response = llm_chain.invoke({"problem": problem})
            solution_text = response['text']

            audio_base64 = text_to_speech(solution_text)  # Get Base64 audio
            audio_html = f"""
                <audio controls>
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
            """
            st.success("✅ Solve (Speech) Started!")
            st.markdown(audio_html, unsafe_allow_html=True)

            # 📥 Add a download button for mobile users
            audio_file_name = "solution.mp3"
            with open(audio_file_name, "wb") as file:
                file.write(base64.b64decode(audio_base64))
            
            with open(audio_file_name, "rb") as file:
                st.download_button(
                    label="📥 Download Audio",
                    data=file,
                    file_name="solution.mp3",
                    mime="audio/mp3"
                )
    else:
        st.warning("⚠️ Please enter a valid problem.")

# Footer
st.markdown("<br><p style='text-align:center; font-size:14px;'>🚀 Created with ❤️ by an AI-powered tutor! 📖</p>", unsafe_allow_html=True)
