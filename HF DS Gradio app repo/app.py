import gradio as gr
from guidance_helper import generate_html  # Import the function

with gr.Blocks(fill_height=True) as demo:
    with gr.Sidebar():
        gr.Markdown("# Inference Provider")
        gr.Markdown("This Space showcases the deepseek-ai/DeepSeek-V3-0324 model, served by the novita API.")
        button = gr.LoginButton("Sign in")

    text_input = gr.Textbox(label="Enter a topic")
    html_output = gr.HTML(label="Generated HTML")

    generate_button = gr.Button("Generate HTML")
    generate_button.click(generate_html, inputs=text_input, outputs=html_output)

demo.launch()