from backend.tools.disease_matcher_tool import match_disease_info
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSerializable
import os
from backend.utils.llm import set_llm
from langchain.agents import initialize_agent
from dotenv import load_dotenv


class SYMPTOMTODISEASEAGENT:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """
        You're a helpful assistant. When you get a query, you should be able to find the disease
        which the symptoms are apped to using the tools.
        """
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("human", "{query}")]
        )

        self.agent = initialize_agent(
            tools=[match_disease_info],
            llm=self.llm,
            agent="chat-conversational-react-description",
            verbose=True,
        )

    def invoke(self, query):
        response_str = self.agent.invoke({"input": query, "chat_history": []})
        return response_str["output"]


if __name__ == "__main__":
    load_dotenv()

    llm_instane = set_llm(
        os.getenv("GROQ_API_KEY"), os.getenv("TEST_API_BASE"), os.getenv("TEST_MODEL")
    )

    D = SYMPTOMTODISEASEAGENT(llm_instane)
    print(D.invoke("cough, cold , throat pain, tirdness"))
