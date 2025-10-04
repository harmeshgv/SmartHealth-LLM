import streamlit as st
from backend.utils.llm import set_llm
from backend.agent_orchestrator import AgentOrchestration

st.set_page_config(page_title="Smart Health LLM", page_icon="🩺", layout="centered")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "agent_orchestrator" not in st.session_state:
    st.session_state.agent_orchestrator = None

# Sidebar: API key & settings
cloud_provider_link_dict = {
    "Groq": "https://api.groq.com/openai/v1",
    "GravixLayer": "https://api.gravixlayer.com/v1/inference",
}

with st.sidebar:
    st.header("⚙️ Settings")

    # Cloud provider
    api_cloud_provider_input = st.selectbox(
        "Which cloud provider do you want?", ("Groq", "GravixLayer")
    )

    # Models per provider
    default_models = {
        "Groq": [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "moonshotai/kimi-k2-instruct-0905",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-3.3-70b-versatile",
            "meta-llama/llama-3.1-8b-instant",
        ],
        "GravixLayer": [
            "mistralai/mistral-nemo-instruct-2407",
            "meta-llama/llama-3.2-3b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            "deepseek-ai/deepseek-r1-0528-qwen3-8b",
            "meta-llama/llama-3.2-1b-instruct",
            "microsoft/phi-4",
        ],
    }
    selected_model = st.selectbox(
        "Select model",
        default_models[api_cloud_provider_input],
        index=0,
    )

    # API key input
    api_key_input = st.text_input(
        "Enter your API key",
        type="password",
        placeholder="Paste your API key here",
        key="api_key_input",
    )

    # Submit API key
    if st.button("🔑 Submit API Key"):
        if not api_key_input:
            st.error("⚠️ Please enter a valid API key!")
        else:
            # Pick API base
            api_base = cloud_provider_link_dict[api_cloud_provider_input]

            # Initialize LLM + orchestrator
            try:
                llm = set_llm(
                    api_key=api_key_input, model=selected_model, api_key_base=api_base
                )
                st.session_state.agent_orchestrator = AgentOrchestration(llm)
                st.session_state.api_key = api_key_input
                st.session_state.messages = []  # reset chat
                st.success(f"✅ API key submitted! Using model: {selected_model}")
            except Exception as e:
                st.error(f"❌ Error initializing agent: {str(e)}")

    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# Title
st.title("🤖 Smart Health LLM")

# Show chat history
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("🩺 **How may I assist you today?**")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
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
                    orchestrator = st.session_state.agent_orchestrator

                    # FIX: The main() method returns the formatted result directly
                    answer = orchestrator.main(prompt)

                    # If we get a dictionary (old behavior), extract the result
                    if isinstance(answer, dict):
                        if "formatted_result" in answer:
                            answer = answer["formatted_result"]
                        elif "result" in answer:
                            answer = answer["result"]
                        else:
                            answer = str(answer)

                    # If empty response, provide fallback
                    if not answer or len(answer.strip()) < 10:
                        answer = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question or check your API configuration."

                except Exception as e:
                    st.error(f"Error details: {str(e)}")
                    answer = f"❌ Sorry, I encountered an error while processing your request. Please try again or check the console for details. Error: {str(e)}"

                st.markdown(answer)

        # Save assistant reply
        st.session_state.messages.append({"role": "assistant", "content": answer})
