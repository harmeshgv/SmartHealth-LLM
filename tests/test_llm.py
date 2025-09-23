from backend.utils.llm import set_llm
import pytest
import os
from dotenv import load_dotenv
import streamlit as st


def test_llm_initialization():
    load_dotenv()
    api_key = os.getenv("TEST_API_KEY")
    api_base = os.getenv("TEST_API_BASE")
    model = os.getenv("TEST_MODEL")

    success = set_llm(api_key, api_base, model)

    assert success
