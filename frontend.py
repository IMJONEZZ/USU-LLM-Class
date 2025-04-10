from zenml import step
import torch
from PIL import Image
import gradio as gr
from transformers import LlamaForCausalLM, AutoProcessor


def load_model():
    model_id = "meta-llama/Llama-3.2-1B"
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Check if GPU is available

    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "6GB", "cpu": "12GB"},
    )

    model.tie_weights()  # Tying weights for efficiency
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"Model loaded on: {device}")

    return model, processor


def process_ticket(text):
    model, processor = load_model()

    try:
        prompt = f"<|begin_of_text|>{text}"
        # Process text-only input
        inputs = processor(text=prompt, return_tensors="pt").to(model.device)

        # Generate response (restrict token length for faster output)
        outputs = model.generate(**inputs, max_new_tokens=200)
        # Decode the response from tokens to text
        response = processor.decode(outputs[0], skip_special_tokens=True)
        return response

    except Exception as e:
        print(f"Error processing ticket: {e}")
        return "An error occurred while processing your request."


@step
def create_interface():
    # Create the Gradio interface
    interface = gr.ChatInterface(
        fn=process_ticket,  # Function to process inputs
        title="Text Completion Llama Chatbot",
        description="Let an LLM write some more text based on your prompt.",
    )

    # Launch the interface with debug mode
    # See chatbot at http://127.0.0.1:7860
    print("Launching chatbot!")
    interface.launch()
