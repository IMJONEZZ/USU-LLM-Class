from guidance import models, gen
from guidance import user, assistant, system
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptOptimizer:
    """Class for optimizing prompts without constrained sampling."""

    def __init__(self, model_name="gpt-3.5-turbo", api_key=None):
        """Initialize the PromptOptimizer with the specified model.

        Args:
            model_name (str): The name of the model to use
            api_key (str): The API key for the model
        """
        self.model_name = model_name
        self.api_key = api_key
        self.llm = models.OpenAI(model_name, api_key=api_key)

    def generate_text(self, prompt):
        """Generate a text response to a general prompt.

        Args:
            prompt (str): The user's prompt

        Returns:
            str: The generated text response
        """
        lm = self.llm

        # Use system messaging to establish the context
        with system():
            lm += "You are a helpful assistant that provides informative responses."

        # Add the user's prompt
        with user():
            lm += prompt

        # Capture the assistant's response
        with assistant():
            lm += gen(name="text_output")

        # Return the generated text
        return lm["text_output"]

    def generate_html(self, topic, content_type="article"):
        """Generate HTML content based on the topic and content type.

        Args:
            topic (str): The topic for the HTML content
            content_type (str): The type of content (article or product)

        Returns:
            str: The generated HTML content
        """
        lm = self.llm

        # Use system messaging to establish the context
        with system():
            lm += f"You are a helpful assistant that generates valid HTML content about {topic}."

        # Provide detailed instructions in the user message
        with user():
            lm += self._build_prompt(topic, content_type)

        # Capture the assistant's response
        with assistant():
            lm += gen(name="html_output")

        # Return the generated HTML
        return lm["html_output"]

    def _build_prompt(self, topic, content_type):
        """Build an optimized prompt for HTML generation.

        Args:
            topic (str): The topic for the HTML content
            content_type (str): The type of content

        Returns:
            str: The optimized prompt
        """
        # Base instructions that apply to all content types
        base_instructions = f"""
        Please create an HTML page about {topic}.
        
        Your response should:
        1. Contain valid, well-structured HTML
        2. Include appropriate meta tags
        3. Use semantic HTML elements
        4. Have a clean, professional design
        5. Be responsive and accessible
        
        Only return the HTML code without any explanation or markdown formatting.
        """

        # Content type specific instructions
        if content_type == "article":
            specific_instructions = f"""
            The HTML should be structured as an article with:
            - A clear, descriptive title about {topic}
            - An introduction section that explains what {topic} is
            - At least two main content sections with appropriate headings
            - Paragraphs with meaningful content about {topic}
            - A conclusion or summary section
            - A footer with copyright information
            
            The HTML should be valid and properly indented.
            """
        else:  # product
            specific_instructions = f"""
            The HTML should be structured as a product page with:
            - A product title for {topic}
            - A product image placeholder
            - A detailed description of the {topic} product
            - A features section with at least 3 key features in a list
            - A specifications table with relevant information
            - A pricing section with a prominent price display
            - A clear call-to-action button for purchasing
            - A footer with company information
            
            The HTML should be valid and properly indented.
            """

        # Examples to guide the model (few-shot learning)
        examples = """
        Here are the elements of good HTML structure:
        
        1. Start with <!DOCTYPE html>
        2. Include <html>, <head>, and <body> tags
        3. Use semantic elements like <header>, <main>, <section>, <article>, and <footer>
        4. Include meta tags for character set and viewport
        5. Use CSS for styling (you can use inline styles for this task)
        """

        # Combine all parts into a complete prompt
        full_prompt = f"{base_instructions}\n\n{specific_instructions}\n\n{examples}"
        return full_prompt

    def generate_structured_output(self, topic, output_format="json"):
        """Generate structured output about a topic.

        Args:
            topic (str): The topic to generate information about
            output_format (str): The desired output format

        Returns:
            str: The structured output
        """
        lm = self.llm

        with system():
            lm += f"You are a helpful assistant that provides structured information about {topic}."

        with user():
            lm += self._build_structured_prompt(topic, output_format)

        with assistant():
            lm += gen(name="structured_output")

        return lm["structured_output"]

    def _build_structured_prompt(self, topic, output_format):
        """Build an optimized prompt for generating structured output.

        Args:
            topic (str): The topic to generate information about
            output_format (str): The desired output format

        Returns:
            str: The optimized prompt
        """
        format_instructions = ""
        if output_format == "json":
            format_instructions = f"""
            Provide information about {topic} in JSON format.
            
            Your response should:
            1. Be valid JSON
            2. Include the following keys:
               - "title": A title for {topic}
               - "description": A brief description of {topic}
               - "key_points": An array of at least 3 important points about {topic}
               - "details": An object with additional details about {topic}
            
            Ensure the JSON is properly formatted without any explanation or additional text.
            """
        else:  # xml or other formats
            format_instructions = f"""
            Provide information about {topic} in structured {output_format.upper()} format.
            
            Your response should follow the structure specified and contain only the
            requested format without any explanation or additional text.
            """

        # Examples to guide the model
        examples = ""
        if output_format == "json":
            examples = """
            Example of good JSON structure:
            
            {
              "title": "Coffee",
              "description": "Coffee is a brewed drink prepared from roasted coffee beans.",
              "key_points": [
                "Coffee contains caffeine, a natural stimulant",
                "It is one of the most popular beverages worldwide",
                "There are various methods of brewing coffee"
              ],
              "details": {
                "origin": "Ethiopia",
                "varieties": ["Arabica", "Robusta", "Liberica"],
                "preparation_methods": ["Drip", "French Press", "Espresso"]
              }
            }
            """

        # Combine all parts into a complete prompt
        full_prompt = f"{format_instructions}\n\n{examples}"
        return full_prompt
