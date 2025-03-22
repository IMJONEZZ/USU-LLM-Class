from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Load fine-tuned model from Hugging Face
model_name = "eliasthompson304/llama3_story_generator"
model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)


@app.get("/generate")
def generate_story(character: str, setting: str, genre: str):
    prompt = f"Write a {genre} story featuring {character} in {setting}."
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    output = model.generate(**inputs, max_new_tokens=500)
    story = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"story": story}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
