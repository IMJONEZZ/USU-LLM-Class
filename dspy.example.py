import dspy



# Define a structured class for the email response
class EmailTemplate(dspy.Signature):
    subject: str = dspy.OutputField(desc="The subject line of the email")
    greeting: str = dspy.OutputField(desc="A greeting to the recipient")
    body: str = dspy.OutputField(desc="The main content of the email")
    closing: str = dspy.OutputField(desc="A closing statement")
    signature: str = dspy.OutputField(desc="The sender's name or sign-off")

# Initialize a model (you can swap this with another LLM if needed)
dspy.configure(lm=dspy.LM('ollama_chat/llama3.2:1b'), api_base='http://localhost:11434', api_key='')

# Create a structured prompting module
generate_email = dspy.Predict(EmailTemplate)

# Example usage
query = "Remind my team about the project deadline next Friday."
email_response = generate_email()

# Format the output as an email
formatted_email = f"""
Subject: {email_response.subject}

{email_response.greeting},

{email_response.body}

{email_response.closing},
{email_response.signature}
"""

print(formatted_email)

