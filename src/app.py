from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
from agent import WebAgent
import uuid
import random

app = FastAPI()
web_agent = WebAgent()

model_name = "gpt2-medium"
model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)


@app.get("/ask")
def ask(question: str, session_id: str = None):
    # Create a new session ID if it doesn't exist
    if session_id is None:
        session_id = str(uuid.uuid4())

    # Generate the prompt with web-enhanced information
    sessionid = random.randint(1, 100)
    user_question = input("enter in a prompt (web agent enabled): ")
    prompt = web_agent.build_prompt_with_web(sessionid, user_question)
    print("======= PROMPT =======")
    print(prompt)
    print("======================")
    # Tokenize the prompt and generate a response
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.5,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.3,
    )

    # Decode and clean the generated response
    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Assistant:" in full_output:
        response = full_output.split("Assistant:")[-1].strip()
    else:
        response = full_output.strip()

    # Save the memory of the session
    web_agent.remember(session_id, question, response)

    return {"session_id": session_id, "response": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
