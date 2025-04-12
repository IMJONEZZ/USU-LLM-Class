from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
from agent import WebAgent
import uuid
import random
import gradio as gr
import uvicorn

app = FastAPI()
web_agent = WebAgent()

# Load the model and tokenizer
model_name = "gpt2-medium"
model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Logic function for FastAPI and Gradio
def generate_response(user_question, session_id=None):
    if session_id is None or session_id.strip() == "":
        session_id = str(uuid.uuid4())

    sessionid = random.randint(1, 100)
    prompt = web_agent.build_prompt_with_web(sessionid, user_question)
    print("======= PROMPT =======")
    print(prompt)
    print("======================")

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.5,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.3,
    )

    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Assistant:" in full_output:
        response = full_output.split("Assistant:")[-1].strip()
    else:
        response = full_output.strip()

    web_agent.remember(session_id, user_question, response)
    return response, session_id

# FastAPI endpoint
@app.get("/ask")
def ask(question: str, session_id: str = None):
    response, session_id = generate_response(question, session_id)
    return {"session_id": session_id, "response": response}

# Gradio interface
gradio_interface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your question here...", label="Question"),
        gr.Textbox(lines=1, placeholder="Session ID (optional)", label="Session ID (optional)"),
    ],
    outputs=[
        gr.Textbox(label="Response"),
        gr.Textbox(label="Session ID"),
    ],
    title="Web-Enhanced LLM Agent",
    description="Ask a question and get a response with web-enhanced context.",
)

# Optional: Launch Gradio when this file runs directly
if __name__ == "__main__":
    import threading

    def launch_gradio():
        gradio_interface.launch(server_name="0.0.0.0", server_port=7860, share=False)

    # Run Gradio in a separate thread so FastAPI can also start
    threading.Thread(target=launch_gradio).start()
    uvicorn.run(app, host="0.0.0.0", port=8001)
