from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

model_name = "eliasthompson304/llama3_story_generator"
model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)


@app.get("/generate", response_class=HTMLResponse)
def generate_story(character: str, setting: str, genre: str):
    prompt = f"A detailed {genre} story about a person named {character} in {setting}. Ensure to make a detailed introduction and resolving conclusion."
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    output = model.generate(
        **inputs,
        max_new_tokens=10000,
        temperature=0.7,
        top_k=50,
        repetition_penalty=1.2,
        top_p=0.9,
    )
    story = tokenizer.decode(output[0], skip_special_tokens=True)

    html_content = f"""
    <html>
        <head>
            <title>Generated Story</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }}
                .container {{
                    max-width: 700px;
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                    text-align: left;
                    line-height: 1.6;
                    word-wrap: break-word;
                    overflow-wrap: break-word;
                }}
                h1 {{
                    color: #007bff;
                    font-size: 24px;
                    margin-bottom: 10px;
                }}
                p {{
                    font-size: 18px;
                    white-space: pre-wrap;  /* Ensures line breaks are respected */
                    overflow-wrap: break-word;  /* Prevents text from overflowing */
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Generated Story</h1>
                <p>{story}</p>
            </div>
        </body>
    </html>
    """
    return html_content


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
