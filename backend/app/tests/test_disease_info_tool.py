import os
import pytest
from app.config import settings
from app.tools.medical.disease_info_tool import DiseaseInfoRetrieverTool



@pytest.mark.asyncio
async def test_disease_info_tool():
    # Ensure CSV exists
    assert os.path.exists(settings.DISEASE_INFO_PATH), "CSV file missing!"

    tool = DiseaseInfoRetrieverTool()   # <-- NO csv_path argument

    result = await tool.run("dengue")

    assert "info" in result
    assert isinstance(result["info"], dict)
    assert "disease" in result["info"]
