# from generate_html import template, generator

# def generate_html(prompt: str, seed: int) -> str:
#     student = generator(prompt, seed=seed)
#     html_output = template.render(student=student.dict())
#     return html_output

# if __name__ == "__main__":
#     output = generate_html("Give me a student description", 789001)
#     print(output)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from generate_html import template, generator

# Define a Pydantic model for the input request
class GenerateRequest(BaseModel):
    prompt: str
    seed: int

# Initialize FastAPI app
app = FastAPI()

@app.post("/generate_html/")
async def generate_html_endpoint(request: GenerateRequest):
    try:
        print('generating student profile')
        # Generate the student profile based on the prompt and seed
        student = generator(request.prompt, seed=request.seed)
        html_output = template.render(student=student.dict())
        return {"html": html_output}
    except Exception as e:
        # Handle errors gracefully
        print('failed')
        raise HTTPException(status_code=500, detail=str(e))

# Run the server (if running directly)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
