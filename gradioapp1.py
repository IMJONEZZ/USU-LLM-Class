import gradio as gr


# Define the chatbot response function
def chatbot_response(message):
    """Simulate a chatbot response."""
    if not message:
        return "Error: No message received."

    # Simulate model response (you would normally use an API here)
    return f"Response to: {message}"


with gr.Blocks(fill_height=True) as demo:
    with gr.Sidebar():
        gr.Markdown("# Inference Provider")
        gr.Markdown("This Space showcases the deepseek-ai/DeepSeek-R1 model.")
        button = gr.LoginButton("Sign in")  # Allow users to sign in

    # Define the Gradio interface directly inside the Blocks context
    chatbot = gr.Interface(
        fn=chatbot_response,  # Your custom chatbot function
        inputs="text",  # User input type
        outputs="text",  # Response output type
    )

    chatbot.render()  # This renders the chatbot interface inside the Blocks layout

demo.launch()  # Launch the Space
