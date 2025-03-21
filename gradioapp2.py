import gradio as gr

with gr.Blocks(fill_height=True) as demo:
    with gr.Sidebar():
        gr.Markdown("# Inference Provider")
        gr.Markdown(
            "This Space showcases the deepseek-ai/DeepSeek-R1 model, served by the together API. Sign in with your Hugging Face account to use this API."
        )
        button = gr.LoginButton("Sign in")
    gr.load("models/deepseek-ai/DeepSeek-R1", accept_token=button, provider="together")

demo.launch()
