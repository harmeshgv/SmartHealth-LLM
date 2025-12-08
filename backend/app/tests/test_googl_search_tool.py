import pytest
from app.tools.internet.google_search_tool import GoogleSearchTool


@pytest.mark.asyncio
async def test_google_search_tool():
    tool = GoogleSearchTool()
    output = await tool.run("India Groq LPU")

    assert isinstance(output, dict)
    assert "error" in output or "result" in output

    if "result" in output:
        assert isinstance(output["result"], str)
