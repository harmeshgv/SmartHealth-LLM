import os
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.tools import tool

# Load environment variables
load_dotenv()
serper_api_key = os.getenv("SERPER_API_KEY")

# Initialize search tool
search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)


@tool
def google_search(query: str) -> str:
    """
    Retrievs information through google search
    """
    result = search.run(query)
    return result
