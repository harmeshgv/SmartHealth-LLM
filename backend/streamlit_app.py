import streamlit as st
from agents.decider_agent import DECIDERAGENT

st.set_page_config(page_title="Smart Health LLM", page_icon="🩺", layout="centered")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "agent" not in st.session_state:
    st.session_state.agent = None

# Sidebar: API key & clear chat
# Sidebar: API key input and submit
with st.sidebar:
    st.header("⚙️ Settings")
    
    api_key_input = st.text_input(
        "Enter your API key",
        type="password",
        placeholder="Paste your API key here",
        key="api_key_input"
    )

    # Send button
    if st.button("🔑 Submit API Key"):
        if not api_key_input:
            st.error("⚠️ Please enter a valid API key!")
        else:
            st.session_state.api_key = api_key_input
            st.session_state.agent = DECIDERAGENT(api_key=api_key_input)
            st.session_state.messages = []  # reset chat on new key
            st.success("API key submitted successfully!")
            st.rerun()  # refresh app to use new key

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()


# Initialize agent if not already
if not st.session_state.agent and st.session_state.api_key:
    st.session_state.agent = DECIDERAGENT(api_key=st.session_state.api_key)

# Title
st.title("🤖 Smart Health LLM")

# Show chat history
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("🩺 **How may I assist you today?**")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input (fixed at bottom)
if prompt := st.chat_input("Type your symptoms or questions..."):
    if not st.session_state.api_key:
        st.error("⚠️ Please enter your API key in the sidebar to continue.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("⏳ Thinking..."):
                try:
                    answer = st.session_state.agent.main(prompt)
                except Exception as e:
                    answer = f"❌ Error: {str(e)}"
                st.markdown(answer)

        # Save assistant reply
        st.session_state.messages.append({"role": "assistant", "content": answer})
