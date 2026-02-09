# tests/backend/tools/test_skin_disease_prediction_tool.py
import pytest
from unittest.mock import patch, MagicMock, ANY
import logging
from PIL import Image
import io
import pytest_asyncio
import unittest # Import unittest

from app.tools.ml.skin_disease_prediction_tool import SkinDiseasePredictionTool
from app.config import settings

@pytest.fixture(autouse=True)
def cap_log(caplog):
    caplog.set_level(logging.WARNING)

@pytest.fixture
def mock_torch_dependencies():
    """Mocks torch, torchvision.models, and torch.nn."""
    with patch('app.tools.ml.skin_disease_prediction_tool.torch') as mock_torch, \
         patch('app.tools.ml.skin_disease_prediction_tool.models') as mock_models, \
         patch('app.tools.ml.skin_disease_prediction_tool.nn') as mock_nn, \
         patch('app.tools.ml.skin_disease_prediction_tool.transforms') as mock_transforms_module:
        
        # Mock CUDA availability to ensure CPU is used for tests
        mock_torch.cuda.is_available.return_value = False
        
        mock_models.densenet121.return_value = MagicMock()
        # Configure the mock to simulate the 'to' method for device placement
        mock_models.densenet121.return_value.to.return_value = mock_models.densenet121.return_value
        
        mock_torch.load.return_value = MagicMock()
        
        mock_model_instance = mock_models.densenet121.return_value
        mock_model_instance.classifier.in_features = 1024
        mock_nn.Linear.return_value = MagicMock()

        mock_torch.no_grad.return_value.__enter__.return_value = None
        mock_torch.no_grad.return_value.__exit__.return_value = None

        mock_torch.max.return_value = (MagicMock(), MagicMock(item=MagicMock(return_value=0)))

        def mock_compose_side_effect(image_input):
            mock_tensor = MagicMock()
            mock_tensor.unsqueeze.return_value.to.return_value = mock_tensor # Simulate .to(device) call
            return mock_tensor

        mock_transforms_module.Compose.return_value = MagicMock(side_effect=mock_compose_side_effect)

        yield mock_torch, mock_models, mock_nn, mock_transforms_module

@pytest.fixture
def mock_pil_image():
    mock_img = MagicMock(spec=Image.Image)
    mock_img.convert.return_value = mock_img
    mock_img.size = (224, 224)
    return mock_img

@pytest.fixture
def skin_predictor_tool_instance(mock_torch_dependencies):
    # Patching os.path.exists to avoid FileNotFoundError on labels.json
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        with patch('builtins.open', unittest.mock.mock_open(read_data='["class1", "class2"]')):
            return SkinDiseasePredictionTool()


def test_tool_initialization(skin_predictor_tool_instance, mock_torch_dependencies):
    """Test that the tool initializes correctly."""
    mock_torch, mock_models, mock_nn, mock_transforms = mock_torch_dependencies
    
    assert skin_predictor_tool_instance.name == "skin_disease_predictor"
    
    mock_models.densenet121.assert_called_once_with(weights=None)
    mock_nn.Linear.assert_called_once_with(ANY, len(skin_predictor_tool_instance.class_names))
    # Assert that the model is loaded onto the device the tool has chosen
    mock_torch.load.assert_called_once_with(skin_predictor_tool_instance.model_path, map_location=skin_predictor_tool_instance.device)
    assert skin_predictor_tool_instance.model.eval.called
    mock_transforms.Compose.assert_called_once()

def test_tool_initialization_exception():
    """Test error handling during tool initialization."""
    with patch('app.tools.ml.skin_disease_prediction_tool.torch.load', side_effect=Exception("Model load error")):
        with pytest.raises(RuntimeError, match="Failed to load skin model: Model load error"):
            with patch('os.path.exists') as mock_exists:
                mock_exists.return_value = True
                with patch('builtins.open', unittest.mock.mock_open(read_data='["class1", "class2"]')):
                    SkinDiseasePredictionTool()

@pytest.mark.asyncio
async def test_run_with_valid_image(skin_predictor_tool_instance, mock_pil_image, mock_torch_dependencies):
    """Test successful prediction with a valid PIL Image."""
    mock_torch, _, _, mock_transforms = mock_torch_dependencies
    
    # Configure mock_torch.max to return class 0
    mock_torch.max.return_value = (MagicMock(), MagicMock(item=MagicMock(return_value=0)))

    result = await skin_predictor_tool_instance.run(image=mock_pil_image)
    
    assert "predicted_class" in result
    assert result["predicted_class"] == skin_predictor_tool_instance.class_names[0]
    assert "error" not in result
    skin_predictor_tool_instance.model.assert_called_once_with(ANY)
    # The transform itself is mocked, we just need to ensure it was called.
    mock_transforms.Compose.return_value.assert_called_once_with(mock_pil_image)

@pytest.mark.asyncio
async def test_run_with_invalid_input_type(skin_predictor_tool_instance):
    result = await skin_predictor_tool_instance.run(image="not_an_image_path")
    assert "error" in result
    assert "Input must be a PIL Image" in result["error"]
    assert "predicted_class" not in result

@pytest.mark.asyncio
async def test_run_prediction_exception_handling(skin_predictor_tool_instance):
    skin_predictor_tool_instance.model.side_effect = Exception("Prediction failed")
    
    result = await skin_predictor_tool_instance.run(image=MagicMock(spec=Image.Image, size=(224,224)))
    
    assert "error" in result
    assert "Prediction failed" in result["error"]
    assert "predicted_class" not in result
