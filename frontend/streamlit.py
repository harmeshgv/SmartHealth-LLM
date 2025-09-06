import streamlit as st
import requests

st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("🤖 Chatbot Interface")

# Keep chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Type your message here..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send to FastAPI backend
    try:
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"query": prompt}
        )
        reply = response.json().get("answer", "No answer returned.")
    except Exception as e:
        reply = f"Error connecting to backend: {e}"

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Display assistant reply
    with st.chat_message("assistant"):
        st.markdown(reply)
