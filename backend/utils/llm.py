from langchain_openai import ChatOpenAI


def set_llm(api_key=None, api_key_base=None, model="gpt-3.5-turbo"):
    """
    Initialize LLM and store it in st.session_state.llm_instance.
    Returns True if successful.
    """
    if not api_key:
        raise ValueError("⚠️ API key not provided!")

    # Initialize LLM and store in session
    return ChatOpenAI(
        model_name=model,
        temperature=0,
        openai_api_key=api_key,
        openai_api_base=api_key_base,
    )
