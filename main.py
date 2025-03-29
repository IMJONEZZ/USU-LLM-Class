from guidance import models, gen
from guidance import user, assistant, system
import guidance

if __name__ == '__main__':
    llama2 = models.LlamaCpp("E:/DSAI/USU-LLM-Class/Llama-3.2-1B.F16.gguf", echo=False) 

    def generate_html_output(input, model):
        prompt = f"""
            Format the following as html, placing your reponse in the <p> tag in the assistant div:

            <div class="chat">
                <div class="user"><h2>User:</h2> {input}</div>
                <div class="assistant"><h2>Assistant:</h2> <p>"""

        response = model + prompt + gen(stop="</p>")

        return str(response) + "</p></div></div>"
    
    input = "How much wood could a woodchuck chuck if a woodchuck could chuck wood?"
    html = generate_html_output(input, llama2)

    print(html)

