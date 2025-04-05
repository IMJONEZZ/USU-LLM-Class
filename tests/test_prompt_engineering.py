import pytest
import json
import os
from unittest.mock import patch, MagicMock
from prompt_optimizer import PromptOptimizer


class TestPromptOptimizer:
    """Test suite for the PromptOptimizer class."""

    @pytest.fixture
    def mock_openai_model(self):
        """Create a mock for the OpenAI model."""
        with patch("guidance.models.OpenAI") as mock_model:
            # Configure the mock to return a predefined response
            mock_instance = MagicMock()
            mock_instance.__getitem__.return_value = (
                "<html><body>Test HTML</body></html>"
            )
            mock_model.return_value = mock_instance
            yield mock_model

    @pytest.fixture
    def optimizer(self):
        """Create a PromptOptimizer instance for testing."""
        with patch("guidance.models.OpenAI"):
            return PromptOptimizer("test-model", api_key="test-key")

    def test_initialization(self, mock_openai_model):
        """Test that the PromptOptimizer initializes correctly."""
        optimizer = PromptOptimizer("test-model", api_key="test-key")

        # Assert that the OpenAI model was initialized with the correct parameters
        mock_openai_model.assert_called_once_with("test-model", api_key="test-key")

        # Check that the attributes are set correctly
        assert optimizer.model_name == "test-model"
        assert optimizer.api_key == "test-key"

    def test_build_prompt_article(self, optimizer):
        """Test that _build_prompt creates the correct prompt for articles."""
        prompt = optimizer._build_prompt("Python Programming", "article")

        # Check that the prompt contains the topic
        assert "Python Programming" in prompt

        # Check that the prompt contains article-specific instructions
        assert "article" in prompt.lower()
        assert "introduction section" in prompt.lower()
        assert "main content sections" in prompt.lower()

    def test_build_prompt_product(self, optimizer):
        """Test that _build_prompt creates the correct prompt for products."""
        prompt = optimizer._build_prompt("Smart Watch", "product")

        # Check that the prompt contains the topic
        assert "Smart Watch" in prompt

        # Check that the prompt contains product-specific instructions
        assert "product" in prompt.lower()
        assert "product image" in prompt.lower()
        assert "pricing section" in prompt.lower()

    def test_build_structured_prompt_json(self, optimizer):
        """Test that _build_structured_prompt creates the correct prompt for JSON output."""
        prompt = optimizer._build_structured_prompt("Climate Change", "json")

        # Check that the prompt contains the topic
        assert "Climate Change" in prompt

        # Check that the prompt contains JSON-specific instructions
        assert "json" in prompt.lower()
        assert "valid json" in prompt.lower()
        assert "key_points" in prompt.lower()

    @patch("guidance.gen")
    @patch("guidance.system")
    @patch("guidance.user")
    @patch("guidance.assistant")
    def test_generate_html(
        self, mock_assistant, mock_user, mock_system, mock_gen, optimizer
    ):
        """Test the generate_html method."""
        # Configure the mocks to simulate the guidance flow
        mock_gen.return_value = "<html><body>Test HTML</body></html>"

        # Call the method
        result = optimizer.generate_html("Python", "article")

        # Verify the result
        assert result == "<html><body>Test HTML</body></html>"

        # Verify that the guidance functions were called
        mock_system.assert_called_once()
        mock_user.assert_called_once()
        mock_assistant.assert_called_once()
        mock_gen.assert_called_once()

    @patch("guidance.gen")
    @patch("guidance.system")
    @patch("guidance.user")
    @patch("guidance.assistant")
    def test_generate_structured_output(
        self, mock_assistant, mock_user, mock_system, mock_gen, optimizer
    ):
        """Test the generate_structured_output method."""
        # Configure the mocks to simulate the guidance flow
        mock_json = '{"title": "Climate Change", "description": "A test description"}'
        mock_gen.return_value = mock_json

        # Call the method
        result = optimizer.generate_structured_output("Climate Change", "json")

        # Verify the result
        assert result == mock_json

        # Verify that the guidance functions were called
        mock_system.assert_called_once()
        mock_user.assert_called_once()
        mock_assistant.assert_called_once()
        mock_gen.assert_called_once()

    def test_integration_json_output(self):
        """Integration test for generating structured JSON output."""
        # Skip this test if no API key is available
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("No OpenAI API key available")

        # Create a real PromptOptimizer instance
        optimizer = PromptOptimizer(api_key=os.environ.get("OPENAI_API_KEY"))

        # Generate structured output
        result = optimizer.generate_structured_output("Python Programming", "json")

        # Verify that the result is valid JSON
        try:
            parsed = json.loads(result)
            assert isinstance(parsed, dict)
            assert "title" in parsed
            assert "description" in parsed
            assert "key_points" in parsed
            assert isinstance(parsed["key_points"], list)
        except json.JSONDecodeError:
            pytest.fail("Result is not valid JSON")
