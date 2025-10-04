# backend/tools/google_search.py
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

    Input: A search query string.

    Output: A string containing the search results.
    """
    try:
        if not serper_api_key:
            raise ValueError("SERPER_API_KEY is not set in environment variables.")
    except ValueError as e:
        return str(e)

    try:
        if not query or not isinstance(query, str):
            raise ValueError("Invalid query. Please provide a non-empty string.")
    except ValueError as e:
        return str(e)

    try:
        result = search.run(query)
    except Exception as e:
        return f"An error occurred during the search: {str(e)}"

    try:
        if not result:
            return "No results found."
    except ValueError as e:
        return str(e)

    return result
