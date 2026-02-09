import pytest
import os
import shutil
import tempfile
import pandas as pd
from unittest.mock import MagicMock, patch
from scripts.vector_db.build_faiss_db import build_and_save_faiss_db

@pytest.fixture
def dummy_csv():
    """Create a dummy CSV file for testing and return its path."""
    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, "dummy_data.csv")
    data = {
        "disease": ["Common Cold", "Flu"],
        "updated": ["fever,cough", "fever,chills,body ache"]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    yield csv_path
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for the FAISS DB output."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_embeddings():
    """Mock the embeddings model."""
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1, 0.2]] * 2 # Dummy embeddings for 2 documents
    mock.embed_query.return_value = [0.1, 0.2]
    return mock

def test_build_and_save_faiss_db(dummy_csv, temp_output_dir, mock_embeddings):
    """
    Tests the build_and_save_faiss_db function to ensure it creates
    the FAISS index files correctly.
    """
    # We patch FAISS.from_texts because its internal logic is complex and not what we are testing.
    # We are testing that our function processes data and calls the appropriate save method.
    with patch('scripts.vector_db.build_faiss_db.FAISS') as MockFAISS:
        mock_vectorstore = MockFAISS.from_texts.return_value
        
        # Call the function with test data
        build_and_save_faiss_db(
            csv_path=dummy_csv,
            db_output_path=temp_output_dir,
            embeddings=mock_embeddings
        )

        # 1. Assert that FAISS.from_texts was called correctly
        MockFAISS.from_texts.assert_called_once()
        call_args = MockFAISS.from_texts.call_args[0]
        
        # Check the texts and metadatas passed to from_texts
        expected_texts = ["fever; cough", "fever; chills; body ache"]
        expected_metadatas = [{"disease": "Common Cold"}, {"disease": "Flu"}]
        assert call_args[0] == expected_texts
        assert call_args[2] == expected_metadatas
        
        # 2. Assert that save_local was called on the vectorstore instance
        mock_vectorstore.save_local.assert_called_once_with(temp_output_dir)

def test_build_and_save_faiss_db_file_not_found(temp_output_dir, mock_embeddings):
    """
    Tests that the function handles a non-existent CSV file gracefully.
    """
    build_and_save_faiss_db(
        csv_path="non_existent_file.csv",
        db_output_path=temp_output_dir,
        embeddings=mock_embeddings
    )
    # Just asserting that it runs without error. The function prints an error message.
    # A more advanced test could capture stdout to check the message.
    assert not os.path.exists(os.path.join(temp_output_dir, "index.faiss"))
