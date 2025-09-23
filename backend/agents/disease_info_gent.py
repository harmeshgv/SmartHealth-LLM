from langchain_community.vectorstores import FAISS
from backend.tools.disease_info_retriever import match_disease_info
from backend.tools.google_search import google_search
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import initialize_agent
from backend.utils.llm import set_llm
import os


class DISEASEINFOAGENT:
    def __init__(self, llm):
        self.system_prompt = """
        You're a helpful assistant. When you get a query, you should be able to extract
        disease info using any of the tools given.
        """

        self.llm = llm

        # create the prompt template
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("human", "{query}")]
        )

        # create the agent once
        self.agent = initialize_agent(
            tools=[match_disease_info, google_search],
            llm=self.llm,
            agent="chat-conversational-react-description",
            verbose=True,
        )

    def invoke(self, query):
        response = self.agent.invoke({"input": query, "chat_history": []})
        return response["output"]


if __name__ == "__main__":

    llm_instane = set_llm(
        os.getenv("GROQ_API_KEY"), os.getenv("TEST_API_BASE"), os.getenv("TEST_MODEL")
    )

    D = DISEASEINFOAGENT(llm_instane)
    print(D.invoke("common cold"))
