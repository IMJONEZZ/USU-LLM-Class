import dspy
from dspy import Tool;
import wikipedia

MAX_SENTENCES = 1000;

lm = dspy.LM('ollama_chat/erg', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

def search_wikipedia(query: str):
    summary = wikipedia.summary(query)
    return summary

rag = dspy.ChainOfThought('context, question -> response')

question=input("Enter your question: ")

response = rag(context=search_wikipedia(question), question=question)

print(response.completions.response)