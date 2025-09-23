import streamlit as st
from backend.agents.decider_agent import DECIDERAGENT
from backend.agents.disease_info_gent import DISEASEINFOAGENT
from backend.agents.symptom_agent import SYMPTOMTODISEASEAGENT
from utils.llm import set_llm

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

cloud_provider_link_dict = {
    "Groq": "https://api.groq.com/openai/v1",
    "GravixLayer": "https://api.gravixlayer.com/v1/inference",
}

with st.sidebar:
    st.header("⚙️ Settings")

    # 1️⃣ Cloud provider selection
    api_cloud_provider_input = st.selectbox(
        "Which cloud provider you want?", ("Groq", "GravixLayer")
    )
    st.write("You selected:", api_cloud_provider_input)

    # 2️⃣ Model selection based on provider
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
            "mistralai/mistral-nemo-instruct-2407",  # most used
            "meta-llama/llama-3.2-3b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            "deepseek-ai/deepseek-r1-0528-qwen3-8b",
            "meta-llama/llama-3.2-1b-instruct",
            "microsoft/phi-4",
            "mistralai/mistral-nemo-instruct-2407",  # least used
        ],
    }
    selected_model = st.selectbox(
        "Select model",
        default_models[api_cloud_provider_input],
        index=0,  # default first model
    )

    # 3️⃣ API key input
    api_key_input = st.text_input(
        "Enter your API key",
        type="password",
        placeholder="Paste your API key here",
        key="api_key_input",
    )

    # 4️⃣ Submit button
    if st.button("🔑 Submit API Key"):
        if not api_key_input:
            st.error("⚠️ Please enter a valid API key!")
        else:
            st.session_state.api_key = api_key_input

            # Initialize LLM instance dynamically
            # Determine API base dynamically
            api_base = None
            if api_cloud_provider_input == "Groq":
                api_base = cloud_provider_link_dict["Groq"]
            elif api_cloud_provider_input == "GravixLayer":
                api_base = cloud_provider_link_dict["GravixLayer"]

            # Initialize LLM instance
            st.session_state.llm = set_llm(
                api_key=api_key_input, model=selected_model, api_key_base=api_base
            )
            st.session_state.decider_agent = DECIDERAGENT(st.session_state.llm)
            st.session_state.symptom_agent = SYMPTOMTODISEASEAGENT(st.session_state.llm)
            st.session_state.disease_info_agent = DISEASEINFOAGENT(st.session_state.llm)

            # Inject LLM into DECIDERAGENT
            st.session_state.messages = []  # reset chat
            if "llm" in st.session_state:
                st.success(f"API key submitted! Using model: {selected_model}")
            st.rerun()

    # 5️⃣ Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()


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
                    print("done")
                    # Step 1: Get decider agent decision
                    decider_output = st.session_state.decider_agent.invoke(prompt)
                    agent_name = decider_output.content
                    print(agent_name)

                    # Step 2: Call the chosen agent
                    if agent_name == "symptom_to_disease":
                        answer1 = st.session_state.symptom_agent.invoke(prompt)
                        # If you want to enrich with disease info:
                        disease_name = answer1.split(",")[0]  # first disease
                        answer2 = st.session_state.disease_info_agent.invoke(
                            disease_name
                        )
                        answer = answer1 + "\n\n" + answer2
                    elif agent_name == "disease_info":
                        answer = st.session_state.disease_info_agent.invoke(prompt)
                        print(answer)
                    else:
                        answer = "❌ Could not determine which agent to use."

                except Exception as e:
                    answer = f"❌ Error: {str(e)}"

                st.markdown(answer)

        # Save assistant reply
        st.session_state.messages.append({"role": "assistant", "content": answer})
