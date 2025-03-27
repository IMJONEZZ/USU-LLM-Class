import guidance
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Hugging Face model
model_name = "deepseek-ai/DeepSeek-V3-0324"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Set up Guidance with the Transformers model
guidance.llm = guidance.models.Transformers(model, tokenizer)

# Define a structured HTML template
html_template = guidance(
    """
    <html>
        <head><title>{{gen 'title'}}</title></head>
        <body>
            <h1>{{gen 'title'}}</h1>
            <p>{{gen 'intro'}}</p>
            <ul>
                {{#each gen 'points' stop="</ul>"}}
                    <li>{{this}}</li>
                {{/each}}
            </ul>
        </body>
    </html>
    """
)

# Function to generate constrained HTML output
def generate_html(prompt):
    return html_template(title=prompt)