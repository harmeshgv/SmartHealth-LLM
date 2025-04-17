import streamlit as st
import time

# Custom Styling
st.markdown(
    """
    <style>
        body {background-color: #f4f4f4;}
        .main {background: white; padding: 2rem; border-radius: 15px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);}
        h1 {color: #ff4c4c; text-align: center;}
        .stTextInput {border-radius: 10px;}
        .stButton>button {background-color: #ff4c4c; color: white; padding: 10px 20px; border-radius: 10px; font-size: 16px;}
        .stButton>button:hover {background-color: #ff3333;}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("🩺 Disease & Symptom Chatbot")
st.write("💡 Ask me about symptoms and diseases!")

# User input
user_input = st.text_input("🔍 Enter your symptoms:", placeholder="e.g., fever, cough, headache...")

if st.button("🔎 Get Diagnosis"):
    with st.spinner("🤖 Processing your symptoms..."):
        time.sleep(2)  # Simulate processing time
        response = "⚠️ This is a placeholder response. The actual model will provide answers here."
    st.success("✅ Chatbot Response:")
    st.write(response)

st.markdown("</div>", unsafe_allow_html=True)

# Run this with: streamlit run app.py
