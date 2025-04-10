from unittest.mock import patch, MagicMock
from frontend import process_ticket


@patch("frontend.load_model")
def test_process_ticket_text_only(mock_load_model):
    # Mock the model and processor
    mock_model = MagicMock()
    mock_processor = MagicMock()

    # Mock processor output with a .to() method
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs  # Simulate .to() returning itself

    # Setup mock behavior
    mock_processor.return_value = mock_inputs
    mock_model.device = "cpu"
    mock_model.generate.return_value = [["fake_token_ids"]]
    mock_processor.decode.return_value = "Generated response"

    mock_load_model.return_value = (mock_model, mock_processor)

    # Call the function
    result = process_ticket("Hello, world!")

    # Assertions
    assert isinstance(result, str)
    assert "Generated" in result
