# tests/backend/tools/test_skin_disease_prediction_tool.py
import pytest
from unittest.mock import patch, MagicMock, ANY
import logging
from PIL import Image
import io
import pytest_asyncio

from backend.tools.skin_disease_prediction_tool import SkinDiseasePredictionTool
from backend.config import settings

@pytest.fixture(autouse=True)
def cap_log(caplog):
    caplog.set_level(logging.WARNING)

@pytest.fixture
def mock_torch_dependencies():
    """Mocks torch, torchvision.models, and torch.nn."""
    with patch('backend.tools.skin_disease_prediction_tool.torch') as mock_torch, \
         patch('backend.tools.skin_disease_prediction_tool.models') as mock_models, \
         patch('backend.tools.skin_disease_prediction_tool.nn') as mock_nn, \
         patch('backend.tools.skin_disease_prediction_tool.transforms') as mock_transforms_module:
        
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
    return SkinDiseasePredictionTool(
        model_path=settings.DISEASE_PREDICTION_MODEL,
        class_names=settings.SKIN_DISEASE_CLASS_NAMES
    )

def test_tool_initialization(skin_predictor_tool_instance, mock_torch_dependencies):
    """Test that the tool initializes correctly."""
    mock_torch, mock_models, mock_nn, mock_transforms = mock_torch_dependencies
    
    assert skin_predictor_tool_instance.name == "Skin Disease Predictor"
    assert "Predicts the type of skin disease" in skin_predictor_tool_instance.description
    
    mock_models.densenet121.assert_called_once_with(weights=None)
    mock_nn.Linear.assert_called_once_with(ANY, len(settings.SKIN_DISEASE_CLASS_NAMES))
    # Assert that the model is loaded onto the device the tool has chosen
    mock_torch.load.assert_called_once_with(settings.DISEASE_PREDICTION_MODEL, map_location=skin_predictor_tool_instance.device)
    assert skin_predictor_tool_instance.model.eval.called
    mock_transforms.Compose.assert_called_once()

def test_tool_initialization_exception():
    """Test error handling during tool initialization."""
    with patch('backend.tools.skin_disease_prediction_tool.torch.load', side_effect=Exception("Model load error")):
        with pytest.raises(Exception, match="Model load error"):
            SkinDiseasePredictionTool(
                model_path="invalid_path.pth",
                class_names=["class1"]
            )

@pytest.mark.asyncio
async def test_execute_with_valid_image(skin_predictor_tool_instance, mock_pil_image, mock_torch_dependencies):
    """Test successful prediction with a valid PIL Image."""
    mock_torch, _, _, mock_transforms = mock_torch_dependencies
    
    # Configure mock_torch.max to return class 0
    mock_torch.max.return_value = (MagicMock(), MagicMock(item=MagicMock(return_value=0)))

    result = await skin_predictor_tool_instance.execute(image=mock_pil_image)
    
    assert "predicted_class" in result
    assert result["predicted_class"] == settings.SKIN_DISEASE_CLASS_NAMES[0]
    assert "error" not in result
    skin_predictor_tool_instance.model.assert_called_once_with(ANY)
    # The transform itself is mocked, we just need to ensure it was called.
    mock_transforms.Compose.return_value.assert_called_once_with(mock_pil_image)

@pytest.mark.asyncio
async def test_execute_with_invalid_input_type(skin_predictor_tool_instance):
    result = await skin_predictor_tool_instance.execute(image="not_an_image_path") # Await the async method
    assert "error" in result
    assert "Input must be a PIL Image object." in result["error"]
    assert "predicted_class" not in result

@pytest.mark.asyncio
async def test_execute_prediction_exception_handling(skin_predictor_tool_instance):
    skin_predictor_tool_instance.model.side_effect = Exception("Prediction failed")
    
    result = await skin_predictor_tool_instance.execute(image=MagicMock(spec=Image.Image, size=(224,224))) # Await the async method
    
    assert "error" in result
    assert "Prediction failed" in result["error"]
    assert "predicted_class" not in result
